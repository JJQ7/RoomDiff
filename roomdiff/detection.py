"""Detection channel: detect every object in both recordings, then associate the two object
tables; what fails to associate is a change candidate.

The geometry channel (M3) and the appearance channel both need a *difference* first (free
space, feature dissimilarity on co-visible keyframes), so an object whose difference they
miss never becomes a candidate. This stage works the other way round, like the perception
layer of SuperMap (detector + segmenter per keyframe, then object association), with an
open vocabulary instead of a fixed prompt list:

1. Vocabulary: Qwen3-VL (`semantics` model) lists the objects it sees in `vocab_frames`
   keyframes per recording (spread over time, the sharpest of each time segment). The
   phrases are normalised (`roomdiff.semantics.normalise_label`: lowercase, no articles, cut
   at a preposition, <= 3 words), parts of the room (`is_background`: floor, wall,
   ceiling, door, window, ...) and generic words (object, item, furniture, person, ...) are
   dropped, duplicates (same words up to a plural) are merged; the `max_prompts` most
   frequently named ones are kept, plus `extra_prompts`. The VLM is freed before SAM 3
   loads (16.7 GiB).
2. Detections: SAM 3 (`recheck.model`) with every phrase on every keyframe (the
   full-resolution JPEG, masks at the recon resolution); detections from `score_min` with
   `min_mask_px` .. `max_mask_frac` of the image. Within a keyframe, detections of
   different phrases whose masks overlap at IoU > `merge_iou` are one detection voting for
   every phrase (votes = SAM 3 score per phrase). The raw detections are kept as COCO RLE
   and decoded one keyframe at a time; a merged detection keeps its mask as the crop
   inside its box (`BoxMask`), so memory grows with the masked area, not with the number
   of detections times the image size (store pairs: 25 000 raw detections).
3. Objects per recording: each detection's mask (eroded by `lift_erode_px`) is lifted to
   3D (valid depth, 1 cm voxels, outliers removed) and bounded to its main connected
   part(s) (cells of `instances.lift_cell_m` linked within `lift_link_m`; parts holding at
   least `MAIN_PART_SHARE` of the largest one), so a mask that bled onto the background does
   not carry it along. Detections are clustered across keyframes (union-find): more than
   `cluster_frac` of the points of one within `cluster_dist` of the other *and* a shared
   phrase, or more than `cluster_frac_any` of the points of *each* within `cluster_dist` of
   the other whatever the phrases (a book lying on a table is not merged with the table:
   most of its points lie on the table, but few of the table's lie on the book). An
   object's points are the union of its detections', minus the points that fewer than
   half of its keyframes able to see them put inside their mask (as in the appearance
   channel); objects need `min_views` keyframes, `min_points` points and an oriented box no
   longer than `max_extent` (walls, floors and room-sized regions are not objects).
   Clusters that fail only the first two tests (with at least `MIN_DETECTION_POINTS`
   points) are kept as *weak* objects: they never become candidates, but they count as
   counterparts in the association (an object that SAM 3 found in one keyframe of the
   other recording is still there; absence needs stronger evidence than presence).
4. Association: every object gets a DINOv3 feature (patch tokens pooled in its masks,
   averaged over its `feature_views` best views, as `roomdiff.instances.describe_features`).
   A before object b and an after object a are at the *same place* if at least
   `same_place_frac` of the points of the smaller one lie within `same_place_dist` of the
   other's, or their box centres are closer than max(`same_place_centre_min`,
   `same_place_centre_rel` x the longest box side of the two); their *labels are
   compatible* if their top phrases are equal or the cosine of their phrase-vote vectors is
   at least `label_cos_min`. An object is *unchanged* if the other recording has an object
   at the same place with a compatible label and a feature cosine of at least
   `same_feature_min` (not one-to-one: two fragments of one sofa may both match it; the
   counterpart may be weak). Among the others (weak objects excluded), same-place pairs are
   candidate in-place *replacements* (greedy one-to-one,
   largest point overlap first; changes.py links them by box overlap like appearance
   pairs); everything else is *unmatched*. A phrase + feature match at different places
   is two unmatched objects (`nearest_compatible` records the would-be move); detection
   candidates are never paired into moves (changes.py, semantics.py): in a shop every
   price tag or bottle has a same-looking counterpart elsewhere.
5. Gate: `observing_views` = keyframes of the other recording in which at least 15 % of
   the object's points project inside the image and are not occluded (5 cm, as the
   appearance channel); `surface_consistent` / `see_through` = shares of the visible
   points over those views whose observed depth is within 5 cm of the point / more than
   5 cm behind it. A replaced or unmatched object becomes a `disappeared` (before) /
   `appeared` (after) candidate only if (`gate_reasons`) the other recording saw its place
   in at least `min_observing_views` keyframes (never seen: dropped, the channel makes no
   unverified regions), its mean SAM 3 score is at least `min_mean_score`, and its top
   label names at most `max_label_count` (not weak) objects of its recording (look-alike
   items, price tags or bottles on a shelf, caused most false candidates). The dropped
   ones are listed with their reasons (association.json `gate`). `see_through` does not
   gate (a flat object on a support reads as "surface still there": laptop 0.01, tissue
   0.00); M5 uses it only to overrule a judge that calls a candidate unchanged
   (`see_through_override`, roomdiff.semantics).

The instance stage appends the candidates that no geometry or appearance instance covers
(`load_detection_instances`, `source: "detection"`); changes.py scales their confidence by
the mean SAM 3 score of their views (`detection_support`), pairs them in place only with
each other and keeps them out of the moved matching, and M5 does not re-pair them. Writes
`<out>/detection/` (docs/interfaces.md, "Detection channel").
"""

from __future__ import annotations

import gc
import json
import re
import resource
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from roomdiff.appearance import (
    CONSENSUS_MIN_FRAMES,
    CONSENSUS_SHARE,
    PLACE_DEPTH_TOL,
    depth_tolerance,
    free_evidence,
)
from roomdiff.geometry import floor_obb, unproject_depth
from roomdiff.instances import (
    Instance,
    View,
    clean_points,
    decode_rle,
    encode_rle,
    mask_box,
    near_fraction,
    pooled_feature,
)
from roomdiff.models import Featurizer, TextSegmenter
from roomdiff.paths import RECORDINGS, OutputPaths
from roomdiff.projection import MIN_VISIBLE_FRAC, project_to_pixels, score_views, subsample
from roomdiff.recon import load_recon

KINDS = ("disappeared", "appeared")  # candidate kind of an object owned by before / after
DECISIONS = ("unchanged", "replaced", "unmatched")
# why a replaced / unmatched object is not a candidate (`gate_reasons`)
GATE_REASONS = ("no_observing_views", "few_observing_views", "low_score", "common_label")
GATE_PARAMS = ("min_observing_views", "min_mean_score", "max_label_count")
MAIN_PART_SHARE = 0.5  # lifting keeps the 3D parts holding this share of the largest one
MIN_LIFT_PIXELS = 30  # fewer valid pixels after the erosion: the plain mask is lifted
MIN_DETECTION_POINTS = 20  # a lifted detection with fewer points is not clustered
OVERLAP_QUERY = 400  # points per detection queried in the cluster overlap test
MAX_RESIDENT_GIB = 4.0  # refuse to load the next model while more is still on the GPU
CONSENSUS_REL, CONSENSUS_ABS = 0.03, 0.05  # depth tolerance of the consensus vote (appearance)
# words that name no object kind (or none that can change): never a prompt
GENERIC_NOUNS = {"object", "item", "thing", "stuff", "furniture", "decoration", "decor",
                 "clutter", "misc", "miscellaneous", "equipment", "supply", "accessory",
                 "person", "people", "man", "woman", "child", "kid", "hand", "arm", "leg",
                 "foot", "finger", "body", "space", "area", "scene", "view", "photo",
                 "image", "light", "sunlight", "daylight",
                 # parts of an outdoor place
                 "sky", "ground", "road", "street", "sidewalk", "pavement", "grass"}
# the place may be a room, a shop or a street (the earlier "This is a photo of a room" got
# no objects at all on the SceneDiff street pairs); the text is part of the cache key
VOCAB_PROMPT = (
    "This is a photo of a place (a room, a shop, a street, ...). List the distinct physical "
    "objects you can see in it: furniture (for example chair, table, sofa, shelf, cabinet, "
    "bed) and movable things (for example bag, box, bottle, cup, book, pillow, plant, lamp, "
    "clothes, toy). Name each kind of object once, as a person would call it, with a short "
    "lowercase noun phrase of 1-2 words (for example 'office chair', 'water bottle'); "
    "include small objects. Do not list parts of the place (floor, wall, ceiling, door, "
    "window, ground, road, sky), surfaces, materials, colours, people or reflections.\n"
    'Answer with JSON only: {"objects": ["...", "..."]}')


# ------------------------------------------------------------------ vocabulary


def normalise_phrase(text: Any) -> str | None:
    """Prompt phrase of one VLM entry (`semantics.normalise_label`), or None if it names
    part of the room, a generic word or nothing."""
    from roomdiff.semantics import head_noun, is_background, normalise_label

    phrase = normalise_label(text)
    if phrase == "object" or not re.search(r"[a-z]", phrase):
        return None
    if is_background(phrase) or head_noun(phrase) in GENERIC_NOUNS \
            or phrase.split()[-1] in GENERIC_NOUNS:
        return None
    return phrase


def phrase_key(phrase: str) -> str:
    """De-duplication key: the words with the last one singularised ("water bottles" and
    "water bottle" are one phrase)."""
    from roomdiff.semantics import head_noun

    words = phrase.split()
    return " ".join([*words[:-1], head_noun(phrase)])


def parse_objects(answer: str) -> list[str]:
    """Object phrases of a vocabulary answer: the `objects` list of the first JSON object,
    else the quoted strings of the list after the "objects" key (a reply cut off by the
    token limit, e.g. after the model started repeating itself), else []."""
    from roomdiff.vlm import extract_json

    data = extract_json(answer)
    if isinstance(data, dict) and isinstance(data.get("objects"), list):
        return [str(x) for x in data["objects"] if isinstance(x, str | int | float)]
    match = re.search(r'"objects"\s*:\s*\[', answer)
    if match is None:
        return []
    return re.findall(r'"((?:[^"\\\n]|\\.){1,60})"', answer[match.end():])


def build_vocabulary(
    answers: Sequence[Sequence[str]], max_prompts: int, extra: Sequence[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    """Prompt list from the phrases named per asked keyframe (`answers`): normalised
    (`normalise_phrase`), merged by `phrase_key` (the first spelling is kept), ranked by the
    number of keyframes naming them (then first appearance), the first `max_prompts` kept,
    then the `extra` phrases not yet in the list. Returns the list and a record of the
    dropped and capped phrases."""
    first: dict[str, str] = {}
    counts: Counter[str] = Counter()
    order: dict[str, int] = {}
    dropped: list[str] = []
    for frame_answer in answers:
        seen: set[str] = set()
        for raw in frame_answer:
            phrase = normalise_phrase(raw)
            if phrase is None:
                dropped.append(str(raw))
                continue
            key = phrase_key(phrase)
            first.setdefault(key, phrase)
            order.setdefault(key, len(order))
            if key not in seen:
                counts[key] += 1
                seen.add(key)
    ranked = sorted(first, key=lambda k: (-counts[k], order[k]))
    prompts = [first[k] for k in ranked[:max(0, int(max_prompts))]]
    keys = {phrase_key(p) for p in prompts}
    added = []
    for raw in extra:
        phrase = normalise_phrase(raw)
        if phrase is not None and phrase_key(phrase) not in keys:
            prompts.append(phrase)
            keys.add(phrase_key(phrase))
            added.append(phrase)
    return prompts, {"counts": {first[k]: counts[k] for k in ranked},
                     "capped": [first[k] for k in ranked[max(0, int(max_prompts)):]],
                     "dropped": sorted(set(dropped)), "extra": added}


def vocab_frames(frames: Sequence[int], n: int, sharpness: Sequence[float | None]
                 ) -> list[int]:
    """`n` keyframes spread over time: the recording's keyframes cut into `n` consecutive
    segments, the sharpest of each (the middle one without sharpness values)."""
    frames = list(frames)
    if n <= 0 or not frames:
        return []
    out = []
    for seg in np.array_split(np.arange(len(frames)), min(n, len(frames))):
        values = [sharpness[k] for k in seg]
        if all(v is not None for v in values):
            k = int(seg[int(np.argmax(values))])
        else:
            k = int(seg[len(seg) // 2])
        out.append(int(frames[k]))
    return out


def keyframe_sharpness(paths: OutputPaths, files: Sequence[str]) -> list[float | None]:
    """Sharpness of each keyframe file from keyframes/index.json (None if unknown)."""
    if not paths.keyframes_index.is_file():
        return [None] * len(files)
    index = json.loads(paths.keyframes_index.read_text())
    known = {f["file"]: f.get("sharpness") for rec in RECORDINGS
             for f in (index.get(rec) or {}).get("frames", [])}
    return [known.get(str(f)) for f in files]


def keyframe_image(paths: OutputPaths, recon: Mapping[str, Any], frame: int
                   ) -> tuple[np.ndarray, str]:
    """Full-resolution keyframe (recon image as a fallback) and its name below `<out>`
    (`roomdiff.recheck.keyframe_image`)."""
    from roomdiff.recheck import keyframe_image as image_of

    return image_of(paths, recon, frame)


# ------------------------------------------------------------------ detections


def union_box(a: Sequence[int], b: Sequence[int]) -> tuple[int, int, int, int]:
    """Smallest box (x0, y0, x1, y1, inclusive) holding boxes `a` and `b`."""
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


@dataclass
class BoxMask:
    """A bool mask of an (H, W) image kept as its tight box and the crop inside it (a
    detection covers a small part of the keyframe; the full mask is built on demand)."""

    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive)
    crop: np.ndarray  # (y1 - y0 + 1, x1 - x0 + 1) bool
    shape: tuple[int, int]  # (H, W) of the image

    @classmethod
    def of(cls, mask: np.ndarray) -> BoxMask:
        """The box crop (a copy) of a non-empty full-image mask."""
        x0, y0, x1, y1 = mask_box(mask)
        return cls((x0, y0, x1, y1), mask[y0:y1 + 1, x0:x1 + 1].astype(bool),
                   (int(mask.shape[0]), int(mask.shape[1])))

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.crop))

    def window(self, box: Sequence[int]) -> np.ndarray:
        """The mask inside `box` (x0, y0, x1, y1, inclusive; it must hold this box)."""
        x0, y0, x1, y1 = box
        out = np.zeros((y1 - y0 + 1, x1 - x0 + 1), bool)
        bx0, by0, bx1, by1 = self.box
        out[by0 - y0:by1 - y0 + 1, bx0 - x0:bx1 - x0 + 1] = self.crop
        return out

    def full(self) -> np.ndarray:
        """The (H, W) mask."""
        return self.window((0, 0, self.shape[1] - 1, self.shape[0] - 1))

    def union(self, other: BoxMask) -> BoxMask:
        box = union_box(self.box, other.box)
        return BoxMask(box, self.window(box) | other.window(box), self.shape)

    def iou(self, other: BoxMask) -> float:
        """Mask IoU, computed on the union of the two boxes (0 if the boxes do not touch)."""
        a, b = self.box, other.box
        if min(a[2], b[2]) < max(a[0], b[0]) or min(a[3], b[3]) < max(a[1], b[1]):
            return 0.0
        box = union_box(a, b)
        ca, cb = self.window(box), other.window(box)
        union = np.count_nonzero(ca | cb)
        return float(np.count_nonzero(ca & cb) / union) if union else 0.0


@dataclass
class Detection:
    """One object detection in a keyframe (phrases merged), lifted to 3D."""

    rec: int
    frame: int
    region: BoxMask  # the mask at the recon resolution (box crop)
    score: float  # best SAM 3 score
    votes: dict[str, float]  # phrase -> SAM 3 score
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.region.box

    @property
    def mask(self) -> np.ndarray:
        """The (H, W) mask (built on demand)."""
        return self.region.full()


def keep_detection(mask: np.ndarray, score: float, score_min: float, min_px: int,
                   max_frac: float) -> bool:
    """Score at least `score_min`, at least `min_px` pixels and at most `max_frac` of the
    image."""
    area = int(np.count_nonzero(mask))
    return score >= score_min and area >= min_px and area <= max_frac * mask.size


def merge_frame_detections(
    found: Sequence[tuple[str, np.ndarray, float]], iou: float, rec: int = 0, frame: int = 0,
) -> list[Detection]:
    """Detections of one keyframe `(phrase, mask, score)` merged across phrases: greedily
    by score, a detection whose mask overlaps an earlier (higher-scoring) one at IoU >
    `iou` adds its phrase vote to it (the best score per phrase); the mask is the
    highest-scoring one's (kept as its box crop)."""
    out: list[Detection] = []
    for phrase, mask, score in sorted(found, key=lambda d: -d[2]):
        if not mask.any():
            continue
        region = BoxMask.of(mask)
        hit = next((d for d in out if d.region.iou(region) > iou), None)
        if hit is None:
            out.append(Detection(rec, frame, region, float(score), {phrase: float(score)}))
        else:
            hit.votes[phrase] = max(hit.votes.get(phrase, 0.0), float(score))
    return out


def detect_keyframes(
    segmenter: TextSegmenter, images: Callable[[int], tuple[np.ndarray, str]],
    frames: Sequence[int], prompts: Sequence[str], size: tuple[int, int], score_min: float,
) -> tuple[list[tuple[int, str, float, str]], list[float]]:
    """SAM 3 detections `(frame, phrase, score, mask)` with score >= `score_min` of every
    phrase on every keyframe (`images(frame)` -> image, name), masks at `size` (H, W) as
    COCO RLE counts (only one keyframe's masks are held decoded); also the seconds per
    keyframe."""
    rows: list[tuple[int, str, float, str]] = []
    seconds = []
    for frame in frames:
        image, name = images(int(frame))
        t0 = time.perf_counter()
        found = segmenter.detect(name, image, list(prompts), size=size) if prompts else {}
        seconds.append(time.perf_counter() - t0)
        for phrase in prompts:
            masks, scores = found.get(phrase, (np.zeros((0, *size), bool), np.zeros(0)))
            rows += [(int(frame), phrase, float(s), encode_rle(np.asarray(m, bool))["counts"])
                     for m, s in zip(masks, scores) if float(s) >= score_min]
        del found
    return rows, seconds


# ------------------------------------------------------------------ lifting


def main_parts(points: np.ndarray, cell: float, link: float, share: float = MAIN_PART_SHARE
               ) -> np.ndarray:
    """Which points lie in the main 3D part(s): points binned into `cell` cubes, cells
    linked when their centres are at most `link` apart; the connected parts holding at
    least `share` of the largest part's points are kept."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(points) == 0:
        return np.zeros(0, bool)
    ijk = np.floor(np.asarray(points, np.float64) / cell).astype(np.int64)
    cells, owner = np.unique(ijk, axis=0, return_inverse=True)
    owner = owner.ravel()
    pairs = cKDTree((cells + 0.5) * cell).query_pairs(link + 1e-9, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(cells), len(cells)))
    _, comp = connected_components(graph, directed=False)
    sizes = np.bincount(comp[owner])
    return (sizes >= share * sizes.max())[comp[owner]]


def lift_detection(recon: Mapping[str, Any], det: Detection, erode_px: int, voxel: float,
                   cell: float, link: float, world: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """3D points and colours of a detection: the valid-depth pixels of its mask eroded by
    `erode_px` (the plain mask if fewer than `MIN_LIFT_PIXELS` remain), voxel-downsampled
    with outliers removed (`clean_points`), main part(s) only (`main_parts`). Works inside
    the mask's box (the same pixels, in the same order, as on the full keyframe); `world`:
    the keyframe's `unproject_depth` if already computed."""
    frame = det.frame
    x0, y0, x1, y1 = det.box
    rows, cols = slice(y0, y1 + 1), slice(x0, x1 + 1)
    valid = recon["valid"][frame][rows, cols]
    mask = det.region.crop  # zero outside the box, as the erosion's border value
    if erode_px > 0:
        eroded = ndimage.binary_erosion(mask, iterations=erode_px)
        if np.count_nonzero(eroded & valid) >= MIN_LIFT_PIXELS:
            mask = eroded
    if world is None:
        world = unproject_depth(recon["depth"][frame], recon["K"][frame],
                                recon["cam2world"][frame])
    sel = mask & valid
    pts, colors = clean_points(world[rows, cols][sel], recon["images"][frame][rows, cols][sel],
                               voxel)
    keep = main_parts(pts, cell, link)
    return pts[keep].astype(np.float32), colors[keep]


# ------------------------------------------------------------------ clustering


def cluster_detections(
    points: Sequence[np.ndarray], votes: Sequence[Mapping[str, float]], dist: float,
    frac_label: float, frac_any: float,
) -> list[list[int]]:
    """Union-find groups of detections (one recording) of one object: more than
    `frac_label` of the points of one within `dist` of the other's and a shared phrase, or
    more than `frac_any` of the points of each within `dist` of the other's (any phrases).
    The overlap is measured on a fixed subset of `OVERLAP_QUERY` points per detection."""
    n = len(points)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n == 0:
        return []
    lo = np.array([p.min(0) if len(p) else np.full(3, np.inf) for p in points]) - dist
    hi = np.array([p.max(0) if len(p) else np.full(3, -np.inf) for p in points]) + dist
    touch = np.all((lo[:, None] <= hi[None]) & (lo[None] <= hi[:, None]), axis=-1)
    trees: dict[int, cKDTree] = {}
    queries = [subsample(p, OVERLAP_QUERY) for p in points]

    def frac(a: int, b: int) -> float:
        if b not in trees:
            trees[b] = cKDTree(points[b])
        return float(np.isfinite(trees[b].query(queries[a], distance_upper_bound=dist)[0]).mean())

    for a, b in zip(*np.nonzero(np.triu(touch, 1))):
        a, b = int(a), int(b)
        if root(a) == root(b) or not len(points[a]) or not len(points[b]):
            continue
        fa, fb = frac(a, b), frac(b, a)
        shared = bool(set(votes[a]) & set(votes[b]))
        if min(fa, fb) > frac_any or (shared and max(fa, fb) > frac_label):
            parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def consensus_keep(points: np.ndarray, members: Sequence[Detection], recon: Mapping[str, Any],
                   voxel: float) -> np.ndarray:
    """Which object points at least `CONSENSUS_SHARE` of the member keyframes that see them
    (depth-consistent there, or inside that keyframe's lifted detection) hold in their
    detections (the appearance channel's consensus rule); all points with fewer than
    `CONSENSUS_MIN_FRAMES` keyframes."""
    frames = sorted({d.frame for d in members})
    if len(frames) < CONSENSUS_MIN_FRAMES or len(points) == 0:
        return np.ones(len(points), bool)
    height, width = recon["depth"].shape[1:]
    inside_masks = np.zeros(len(points), np.int64)
    seen = np.zeros(len(points), np.int64)
    for frame in frames:
        held = np.zeros(len(points), bool)
        for d in members:
            if d.frame == frame and len(d.points):
                dist = cKDTree(d.points).query(points, distance_upper_bound=1.5 * voxel)[0]
                held |= np.isfinite(dist)
        col, row, z, inside = project_to_pixels(points, recon["K"][frame],
                                                recon["cam2world"][frame], width, height)
        idx = np.flatnonzero(inside)
        d = recon["depth"][frame][row[idx], col[idx]]
        visible = np.zeros(len(points), bool)
        visible[idx] = recon["valid"][frame][row[idx], col[idx]] & (
            np.abs(d - z[idx]) < depth_tolerance(z[idx], CONSENSUS_REL, CONSENSUS_ABS))
        inside_masks += held
        seen += held | visible
    return inside_masks >= CONSENSUS_SHARE * np.maximum(seen, 1)


@dataclass
class DetView:
    """One keyframe of an object: its detections' masks united (box crop), the best SAM 3
    score."""

    frame: int
    region: BoxMask
    score: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.region.box

    @property
    def mask(self) -> np.ndarray:
        """The (H, W) mask (built on demand)."""
        return self.region.full()


@dataclass
class DetObject:
    """An object of one recording: detections clustered across keyframes."""

    rec: int
    members: list[Detection]
    points: np.ndarray  # (M, 3) float32, 1 cm voxels
    colors: np.ndarray  # (M, 3) uint8
    views: list[DetView]  # one per keyframe (masks united), best SAM 3 score first
    votes: dict[str, float]  # phrase -> summed SAM 3 score over the detections
    consensus_dropped: int = 0
    id: int = 0
    feature: np.ndarray | None = None
    obb: dict[str, Any] = field(default_factory=dict)
    decision: str = "unmatched"
    partner: int | None = None  # matched (unchanged) or replacement object of the other rec
    kind: str | None = None  # candidate kind, None if unchanged
    place: dict[str, Any] = field(default_factory=dict)  # gate evidence (`place_evidence`)
    weak: bool = False  # too few views / points: presence evidence only, never a candidate
    nearest: dict[str, Any] | None = None  # nearest compatible object elsewhere (`associate`)
    label_count: int | None = None  # objects (not weak) of its recording with its top label
    gate: list[str] | None = None  # `gate_reasons` (empty: a candidate; None: not gated)

    @property
    def top_label(self) -> str:
        return max(sorted(self.votes), key=lambda k: self.votes[k]) if self.votes else "object"

    @property
    def mean_score(self) -> float:
        return float(np.mean([v.score for v in self.views])) if self.views else 0.0


def object_views(members: Sequence[Detection]) -> list[DetView]:
    """One view per keyframe: the members' masks united, the best score. Best score
    first."""
    views: dict[int, DetView] = {}
    for d in members:
        if d.frame in views:
            old = views[d.frame]
            views[d.frame] = DetView(d.frame, old.region.union(d.region), max(old.score, d.score))
        else:
            views[d.frame] = DetView(d.frame, d.region, d.score)
    return sorted(views.values(), key=lambda v: (-v.score, v.frame))


def build_objects(
    detections: Sequence[Detection], recon: Mapping[str, Any], params: Mapping[str, Any],
    voxel: float,
) -> tuple[list[DetObject], list[DetObject], list[dict[str, Any]]]:
    """Objects of one recording from its lifted detections (module docstring, step 3), the
    weak objects (clusters with fewer than `min_views` keyframes or `min_points` points:
    presence evidence for the association, never candidates) and the dropped clusters with
    the reason."""
    usable = [d for d in detections if len(d.points) >= MIN_DETECTION_POINTS]
    groups = cluster_detections([d.points for d in usable], [d.votes for d in usable],
                                params["cluster_dist"], params["cluster_frac"],
                                params["cluster_frac_any"])
    objects, weak, dropped = [], [], []
    for group in groups:
        members = [usable[k] for k in group]
        votes: dict[str, float] = {}
        for d in members:
            for phrase, score in d.votes.items():
                votes[phrase] = votes.get(phrase, 0.0) + score
        top = max(sorted(votes), key=lambda k: votes[k])
        n_frames = len({d.frame for d in members})
        record = {"recording": RECORDINGS[members[0].rec], "label": top, "views": n_frames,
                  "detections": len(members)}
        pts, cols = clean_points(np.concatenate([d.points for d in members]),
                                 np.concatenate([d.colors for d in members]), voxel)
        keep = consensus_keep(pts, members, recon, voxel)
        n_dropped = int((~keep).sum())
        pts, cols = pts[keep], cols[keep]
        obb = floor_obb(pts)
        if len(pts) and max(obb["extents"]) > params["max_extent"]:
            dropped.append({**record, "points": len(pts), "extent": max(obb["extents"]),
                            "reason": "larger than max_extent"})
            continue
        obj = DetObject(members[0].rec, members, pts.astype(np.float32), cols,
                        object_views(members), votes, n_dropped, obb=obb)
        reason = f"{n_frames} views < min_views" if n_frames < params["min_views"] else \
            "too few points" if len(pts) < params["min_points"] else None
        if reason is None:
            objects.append(obj)
            continue
        dropped.append({**record, "points": len(pts), "reason": reason})
        if len(pts) >= MIN_DETECTION_POINTS:
            obj.weak = True
            weak.append(obj)
    objects.sort(key=lambda o: -len(o.points))
    weak.sort(key=lambda o: -len(o.points))
    return objects, weak, dropped


# ------------------------------------------------------------------ association


def vote_cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Cosine of two phrase-vote vectors (0 if either is empty)."""
    keys = sorted(set(a) | set(b))
    va = np.array([a.get(k, 0.0) for k in keys])
    vb = np.array([b.get(k, 0.0) for k in keys])
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / norm) if norm > 0 else 0.0


@dataclass
class PairScore:
    """How a before object and an after object relate (`pair_score`)."""

    near: float  # share of the smaller point set within same_place_dist of the other
    centre_dist: float  # distance of the oriented box centres (m)
    centre_limit: float  # max(same_place_centre_min, same_place_centre_rel x longest side)
    label_cos: float
    same_label: bool  # equal top phrases
    feat_cos: float | None

    def same_place(self, params: Mapping[str, Any]) -> bool:
        return self.near >= params["same_place_frac"] or self.centre_dist <= self.centre_limit

    def label_ok(self, params: Mapping[str, Any]) -> bool:
        return self.same_label or self.label_cos >= params["label_cos_min"]

    def unchanged(self, params: Mapping[str, Any]) -> bool:
        return self.same_place(params) and self.label_ok(params) and \
            self.feat_cos is not None and self.feat_cos >= params["same_feature_min"]


def pair_score(b: DetObject, a: DetObject, params: Mapping[str, Any], touch: bool = True
               ) -> PairScore:
    """Place, label and feature relation of a before object `b` and an after object `a`;
    `touch` False: their point boxes are farther apart than `same_place_dist`, so no point
    lies near the other set (`near` 0 without the search)."""
    cb = np.asarray(b.obb["center"], np.float64)
    ca = np.asarray(a.obb["center"], np.float64)
    longest = max(*b.obb["extents"], *a.obb["extents"])
    limit = max(params["same_place_centre_min"], params["same_place_centre_rel"] * longest)
    near = near_fraction(b.points, a.points, params["same_place_dist"]) if touch else 0.0
    feat = None if b.feature is None or a.feature is None else float(b.feature @ a.feature)
    return PairScore(near, float(np.linalg.norm(cb - ca)), float(limit),
                     vote_cosine(b.votes, a.votes), b.top_label == a.top_label, feat)


def _pair_record(b: DetObject, a: DetObject, s: PairScore, params: Mapping[str, Any]
                 ) -> dict[str, Any]:
    return {"before": b.id, "after": a.id, "labels": [b.top_label, a.top_label],
            "weak": [b.weak, a.weak], "near": round(s.near, 3),
            "centre_dist": round(s.centre_dist, 3), "centre_limit": round(s.centre_limit, 3),
            "label_cos": round(s.label_cos, 3),
            "feat_cos": None if s.feat_cos is None else round(s.feat_cos, 4),
            "same_place": s.same_place(params), "label_ok": s.label_ok(params),
            "unchanged": s.unchanged(params)}


def associate(before: Sequence[DetObject], after: Sequence[DetObject],
              params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Decide every object (module docstring, step 4): sets `decision`, `partner` and
    `nearest` on the objects (ids must be set; weak objects count as counterparts only)
    and returns the same-place pair records (at least one side not weak)."""
    reach = 2.0 * params["max_extent"]  # boxes farther apart cannot be at the same place
    gap = params["same_place_dist"]
    lo = [np.array([o.points.min(0) for o in objs]).reshape(-1, 3) - gap / 2
          for objs in (before, after)]
    hi = [np.array([o.points.max(0) for o in objs]).reshape(-1, 3) + gap / 2
          for objs in (before, after)]
    scores: dict[tuple[int, int], PairScore] = {}
    for i, b in enumerate(before):
        for j, a in enumerate(after):
            if b.weak and a.weak:
                continue
            if np.linalg.norm(np.subtract(b.obb["center"], a.obb["center"])) <= reach:
                touch = bool(np.all((lo[0][i] <= hi[1][j]) & (lo[1][j] <= hi[0][i])))
                scores[(i, j)] = pair_score(b, a, params, touch)
    for objs in (before, after):
        for o in objs:
            o.decision, o.partner, o.nearest = "unmatched", None, None
    best: dict[tuple[int, int], float] = {}  # (side, index) -> best unchanged feature cosine
    for (i, j), s in scores.items():
        if s.unchanged(params):
            for side, k, other in ((0, i, after[j]), (1, j, before[i])):
                obj = (before, after)[side][k]
                if s.feat_cos > best.get((side, k), -2.0):
                    best[(side, k)] = s.feat_cos
                    obj.decision, obj.partner = "unchanged", other.id
    pending = [(s.near, -s.centre_dist, i, j) for (i, j), s in scores.items()
               if s.same_place(params) and not before[i].weak and not after[j].weak
               and before[i].decision != "unchanged" and after[j].decision != "unchanged"]
    for _, _, i, j in sorted(pending, reverse=True):
        b, a = before[i], after[j]
        if b.decision == "unmatched" and a.decision == "unmatched":
            b.decision, b.partner = "replaced", a.id
            a.decision, a.partner = "replaced", b.id
    for (i, j), s in scores.items():  # nearest label + feature match elsewhere (a move?)
        if s.same_place(params) or not s.label_ok(params) or s.feat_cos is None or \
                s.feat_cos < params["same_feature_min"]:
            continue
        for obj, other in ((before[i], after[j]), (after[j], before[i])):
            if not obj.weak and not other.weak and obj.decision != "unchanged" and (
                    obj.nearest is None or s.centre_dist < obj.nearest["distance"]):
                obj.nearest = {"id": other.id, "distance": round(s.centre_dist, 3),
                               "feat_cos": round(s.feat_cos, 4)}
    return [_pair_record(before[i], after[j], s, params) for (i, j), s in sorted(scores.items())
            if s.same_place(params)]


def object_feature(obj: DetObject, tokens: Callable[[int], np.ndarray], erode_px: int,
                   n_views: int) -> np.ndarray:
    """L2-normalised mean of the mask-pooled patch tokens of the `n_views` best views
    (`roomdiff.instances.pooled_feature`)."""
    per_view = [pooled_feature(tokens(v.frame), v.mask, erode_px) for v in obj.views[:n_views]]
    mean = np.mean(per_view, 0)
    return (mean / max(float(np.linalg.norm(mean)), 1e-12)).astype(np.float32)


# ------------------------------------------------------------------ gate


def place_evidence(points: np.ndarray, recon: Mapping[str, Any], other: int,
                   grid: Mapping[str, Any] | None) -> dict[str, Any]:
    """The other recording's view of an object's place: `observing_views` (keyframes seeing
    at least `MIN_VISIBLE_FRAC` of the points, not occluded by more than `PLACE_DEPTH_TOL`),
    over those views the shares of the visible points whose observed depth lies within
    the tolerance of the point (`surface_consistent`: a surface is still there) or beyond it
    (`see_through`: the place is empty), the M3 free counts over the points' voxels and the
    nearest camera of the other recording."""
    frames = np.flatnonzero(recon["rec"] == other)
    views = score_views(points, frames, recon["cam2world"], recon["K"], recon["depth"],
                        recon["valid"], PLACE_DEPTH_TOL, on_surface=False)
    observing = [v.frame for v in views if v.visible >= MIN_VISIBLE_FRAC]
    height, width = recon["depth"].shape[1:]
    sample = subsample(points, 2000)
    visible = consistent = through = 0
    for frame in observing:
        col, row, z, inside = project_to_pixels(sample, recon["K"][frame],
                                                recon["cam2world"][frame], width, height)
        idx = np.flatnonzero(inside)
        d = recon["depth"][frame][row[idx], col[idx]].astype(np.float64)
        ok = recon["valid"][frame][row[idx], col[idx]] & (d > 0) & (d >= z[idx] - PLACE_DEPTH_TOL)
        visible += int(ok.sum())
        consistent += int((ok & (np.abs(d - z[idx]) <= PLACE_DEPTH_TOL)).sum())
        through += int((ok & (d > z[idx] + PLACE_DEPTH_TOL)).sum())
    free_rays, free_mean = free_evidence(points, grid, other)
    centres = recon["cam2world"][frames, :3, 3].astype(np.float64)
    distance = float(np.linalg.norm(centres - points.mean(0), axis=1).min()) \
        if len(centres) and len(points) else None
    return {
        "observing_views": len(observing),
        "surface_consistent": round(consistent / visible, 4) if visible else None,
        "see_through": round(through / visible, 4) if visible else None,
        "free_rays": free_rays, "free_frames_mean": free_mean,
        "min_camera_distance": None if distance is None else round(distance, 3),
    }


def gate_reasons(observing_views: int, mean_score: float, label_count: int,
                 params: Mapping[str, Any]) -> list[str]:
    """Why a replaced / unmatched object is no candidate (`GATE_REASONS`; empty: it is
    one): the other recording never saw its place ("no_observing_views") or saw it in
    fewer than `min_observing_views` keyframes ("few_observing_views"), its mean SAM 3
    score is below `min_mean_score` ("low_score"), or its top label names more than
    `max_label_count` (not weak) objects of its recording, itself included
    ("common_label": look-alike items such as price tags or bottles on a shelf)."""
    reasons = []
    if observing_views <= 0:
        reasons.append("no_observing_views")
    elif observing_views < params["min_observing_views"]:
        reasons.append("few_observing_views")
    if mean_score < params["min_mean_score"]:
        reasons.append("low_score")
    if label_count > params["max_label_count"]:
        reasons.append("common_label")
    return reasons


def gate_record(obj: DetObject) -> dict[str, Any]:
    """association.json `gate` entry of an object the gate dropped."""
    return {"id": obj.id, "recording": RECORDINGS[obj.rec], "label": obj.top_label,
            "decision": obj.decision, "partner": obj.partner,
            "observing_views": obj.place.get("observing_views"),
            "mean_score": round(obj.mean_score, 4), "label_count": obj.label_count,
            "see_through": obj.place.get("see_through"), "reasons": obj.gate}


# ------------------------------------------------------------------ instances


def instance_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """M4-style instance evidence of an object record (`roomdiff.instances.instance_record`)
    plus the detection terms under `detection`."""
    place = record["place"]
    return {
        "voxels": 0,
        "volume_m3": round(float(np.prod(record["obb"]["extents"])), 6),
        "free_rays": place["free_rays"],
        "free_frames_mean": place["free_frames_mean"],
        "observing_views": place["observing_views"],
        "occ_points": record["n_points"],
        "min_camera_distance": place["min_camera_distance"],
        "detection": {
            "object": record["id"], "top_label": record["top_label"],
            "label_votes": record["label_votes"], "mean_score": record["mean_score"],
            "n_views": len(record["views"]), "observing_views": place["observing_views"],
            "decision": record["decision"], "partner": record["partner"],
            "surface_consistent": place["surface_consistent"],
            "see_through": place["see_through"]},
    }


def load_objects(out_dir: str | Path) -> tuple[list[dict[str, Any]], dict[int, np.ndarray],
                                               dict[int, np.ndarray],
                                               dict[int, list[tuple[int, dict[str, Any]]]]]:
    """Objects of `<out>/detection/`: records, points / colours and (frame, RLE) masks per
    object id."""
    folder = OutputPaths(out_dir).detection
    records = json.loads((folder / "clusters.json").read_text())["clusters"]
    with np.load(folder / "clusters_points.npz") as f:
        points, colors, owner = f["points"], f["colors"], f["cluster"]
    masks: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    with np.load(folder / "clusters_masks.npz") as f:
        size = [int(s) for s in f["size"]]
        for cid, frame, counts in zip(f["cluster"], f["frame"], f["rle"]):
            masks.setdefault(int(cid), []).append((int(frame), {"size": size,
                                                               "counts": str(counts)}))
    ids = [r["id"] for r in records]
    return (records, {i: points[owner == i] for i in ids}, {i: colors[owner == i] for i in ids},
            masks)


def load_detection_instances(
    out_dir: str | Path, existing: Sequence[Sequence[Instance]], covered_dist: float,
    covered_frac: float,
) -> tuple[list[list[Instance]], dict[str, Any]]:
    """Instances (per recording) from the detection candidates (objects with a `kind`)
    that the `existing` (geometry and appearance) instances of their recording do not
    cover: more than `covered_frac` of the object's points within `covered_dist` of an
    existing instance's points means covered. The new instances (`source` "detection", ids
    still 0) keep the object's views; features and per-keyframe masks are computed by the
    caller. Largest first. Returns them and a meta record."""
    records, points, colors, masks = load_objects(out_dir)
    per_rec: list[list[Instance]] = [[] for _ in RECORDINGS]
    covered, added = [], []
    candidates = [r for r in records if r.get("kind")]
    for record in candidates:
        rec = RECORDINGS.index(record["recording"])
        pts = points[record["id"]]
        overlaps = [(near_fraction_of(pts, i.points, covered_dist), i.id)
                    for i in existing[rec] if len(i.points)]
        best = max(overlaps, default=(0.0, None))
        entry = {"object": record["id"], "recording": record["recording"],
                 "label": record["top_label"], "kind": record["kind"]}
        if best[0] > covered_frac:
            covered.append({**entry, "instance": best[1], "overlap": round(best[0], 3)})
            continue
        boxes = {v["frame"]: v for v in record["views"]}
        views = [View(frame, tuple(boxes[frame]["box"]), decode_rle(rle), boxes[frame]["score"])
                 for frame, rle in masks.get(record["id"], [])]
        views.sort(key=lambda v: -v.score)
        per_rec[rec].append(Instance(rec, record["kind"], [], pts, pts.astype(np.float32),
                                     colors[record["id"]], views, source="detection",
                                     evidence=instance_evidence(record)))
        added.append({**entry, "overlap": round(best[0], 3)})
    for instances in per_rec:
        instances.sort(key=lambda i: -len(i.points))
    return per_rec, {"objects": len(records), "candidates": len(candidates),
                     "covered": covered, "added": added}


def near_fraction_of(points: np.ndarray, other: np.ndarray, dist: float) -> float:
    """Share of `points` within `dist` of `other` (0 if either is empty)."""
    if len(points) == 0 or len(other) == 0:
        return 0.0
    return float(np.isfinite(cKDTree(other).query(points, distance_upper_bound=dist)[0]).mean())


# ------------------------------------------------------------------ stage helpers


def _torch() -> Any:
    torch = sys.modules.get("torch")
    return torch if torch is not None and torch.cuda.is_initialized() else None


def _free_gpu(check: bool = True) -> None:
    """Collect garbage and empty the CUDA cache; with `check`, refuse to go on while more
    than `MAX_RESIDENT_GIB` is still allocated (a model of an earlier step was not freed)."""
    gc.collect()
    torch = _torch()
    if torch is None:
        return
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated() / 2**30
    if check and resident > MAX_RESIDENT_GIB:
        raise RuntimeError(f"{resident:.1f} GiB still allocated on the GPU: refusing to load "
                           "the next model")
    torch.cuda.reset_peak_memory_stats()


def _vram_peak() -> dict[str, float] | None:
    torch = _torch()
    if torch is None:
        return None
    return {"allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


def _vocabulary(paths: OutputPaths, recon: Mapping[str, Any], cfg: Mapping[str, Any],
                params: Mapping[str, Any], vlm: Any, folder: Path
                ) -> tuple[list[str], dict[str, Any]]:
    """Step 1: the prompt list and the vocabulary.json document; the VLM answers are reused
    from an earlier vocabulary.json asked with the same model, prompt and keyframes."""
    sem = cfg["semantics"]
    sharp = keyframe_sharpness(paths, [str(f) for f in recon["files"]])
    asked = {name: vocab_frames(np.flatnonzero(recon["rec"] == r), params["vocab_frames"],
                                [sharp[i] for i in np.flatnonzero(recon["rec"] == r)])
             for r, name in enumerate(RECORDINGS)}
    model = getattr(vlm, "model_id", None) or sem["model"]
    key = {"model": model, "prompt": VOCAB_PROMPT,
           "frames": {n: [str(recon["files"][i]) for i in f] for n, f in asked.items()}}
    file = folder / "vocabulary.json"
    old = json.loads(file.read_text()) if file.is_file() else {}
    cached = old.get("key") == key and vlm is None and \
        all("answer" in a for a in old.get("answers", []))
    if cached:
        answers = old["answers"]
        meta = {"cached": True, "seconds": 0.0, "peak_vram_gb": None}
    else:
        from PIL import Image

        own = vlm is None
        t0 = time.perf_counter()
        if own:
            from roomdiff.vlm import load_vlm

            _free_gpu()
            vlm = load_vlm(sem)
        load_s = time.perf_counter() - t0
        vlm.open_log(folder / "vlm_log.jsonl")
        answers = []
        for name, frames in asked.items():
            for i in frames:
                image, rel = keyframe_image(paths, recon, int(i))
                text = vlm.ask([Image.fromarray(image)], VOCAB_PROMPT, f"vocab:{name}:{i}", [rel],
                               max_new_tokens=params["vocab_max_tokens"])
                answers.append({"recording": name, "frame": int(i),
                                "file": str(recon["files"][i]), "answer": text})
        meta = {"cached": False, "load_s": round(load_s, 2),
                "seconds": round(time.perf_counter() - t0, 2), "calls": vlm.calls,
                "peak_vram_gb": _vram_peak()}
        if own:
            del vlm
            _free_gpu(check=False)
    for a in answers:
        a["objects"] = parse_objects(a["answer"])
    prompts, info = build_vocabulary([a["objects"] for a in answers], params["max_prompts"],
                                     params.get("extra_prompts") or [])
    doc = {"key": key, "answers": answers, "prompts": prompts, **info, "vlm": meta}
    file.write_text(json.dumps(doc, indent=1))
    return prompts, doc


def _detections(paths: OutputPaths, recon: Mapping[str, Any], cfg: Mapping[str, Any],
                params: Mapping[str, Any], prompts: Sequence[str], segmenter: Any, folder: Path
                ) -> tuple[list[list[tuple[int, str, float, str]]], dict[str, Any]]:
    """Step 2: raw SAM 3 detections per recording `(frame, phrase, score, RLE counts)`,
    reused from `detections_<rec>.npz` when model, prompts, score floor and keyframes
    match."""
    height, width = recon["depth"].shape[1:]
    model = getattr(segmenter, "model_id", None) or cfg["recheck"]["model"]
    keys = {name: {"model": model, "prompts": list(prompts), "score_min": params["score_min"],
                   "files": [str(recon["files"][i]) for i in np.flatnonzero(recon["rec"] == r)],
                   "size": [int(height), int(width)]}
            for r, name in enumerate(RECORDINGS)}
    cached = segmenter is None and all(
        (folder / f"detections_{n}.json").is_file() and (folder / f"detections_{n}.npz").is_file()
        and json.loads((folder / f"detections_{n}.json").read_text()) == keys[n]
        for n in RECORDINGS)
    raw: list[list[tuple[int, str, float, str]]] = []
    meta: dict[str, Any] = {"cached": cached}
    if cached:
        for name in RECORDINGS:
            with np.load(folder / f"detections_{name}.npz") as f:
                raw.append([(int(fr), str(p), float(s), str(c))
                            for fr, p, s, c in zip(f["frame"], f["prompt"], f["score"], f["rle"])])
        return raw, meta
    own = segmenter is None
    t0 = time.perf_counter()
    if own:
        from roomdiff.models import Sam3TextSegmenter

        _free_gpu()
        segmenter = Sam3TextSegmenter(cfg["recheck"]["model"], cfg["recheck"]["device"],
                                      params["score_min"])
    meta["load_s"] = round(time.perf_counter() - t0, 2)
    seconds: list[float] = []
    for r, name in enumerate(RECORDINGS):
        rows, secs = detect_keyframes(
            segmenter, lambda i: keyframe_image(paths, recon, i),
            np.flatnonzero(recon["rec"] == r), prompts, (height, width), params["score_min"])
        seconds += secs
        raw.append(rows)
        np.savez_compressed(
            folder / f"detections_{name}.npz",
            frame=np.array([x[0] for x in rows], np.int32),
            prompt=np.array([x[1] for x in rows], dtype=np.str_),
            score=np.array([x[2] for x in rows], np.float32),
            rle=np.array([x[3] for x in rows], dtype=np.str_),
            size=np.array([height, width], np.int64))
        (folder / f"detections_{name}.json").write_text(json.dumps(keys[name]))
    meta.update(seconds=round(time.perf_counter() - t0, 2), images=len(seconds),
                seconds_per_image=round(float(np.mean(seconds)), 3) if seconds else None,
                peak_vram_gb=_vram_peak())
    if own:
        del segmenter
        _free_gpu(check=False)
    return raw, meta


def object_record(obj: DetObject, recon: Mapping[str, Any]) -> dict[str, Any]:
    """clusters.json record of one object."""
    lo, hi = obj.points.min(0), obj.points.max(0)
    return {
        "id": obj.id, "recording": RECORDINGS[obj.rec], "kind": obj.kind, "weak": obj.weak,
        "decision": obj.decision, "partner": obj.partner, "gate": obj.gate,
        "top_label": obj.top_label, "label_count": obj.label_count,
        "label_votes": {k: round(v, 4) for k, v in sorted(obj.votes.items(),
                                                          key=lambda kv: -kv[1])},
        "mean_score": round(obj.mean_score, 4), "n_points": len(obj.points),
        "detections": len(obj.members), "consensus_dropped": obj.consensus_dropped,
        "centroid": obj.points.mean(0).round(4).tolist(),
        "aabb": [lo.round(4).tolist(), hi.round(4).tolist()],
        "extents": (hi - lo).round(4).tolist(), "obb": obj.obb,
        "views": [{"frame": v.frame, "file": str(recon["files"][v.frame]),
                   "box": [int(b) for b in v.box], "mask_box": [int(b) for b in v.box],
                   "score": round(v.score, 4)} for v in obj.views],
        "place": obj.place,
    }


# ------------------------------------------------------------------ stage


def run_detection(
    out_dir: str | Path, cfg: Mapping[str, Any], vlm: Any = None,
    segmenter: TextSegmenter | None = None, featurizer: Featurizer | None = None,
) -> dict[str, Any]:
    """Detection-channel objects and change candidates of `<out>/recon/` (module
    docstring); writes `<out>/detection/`. Qwen3-VL (`semantics`), SAM 3 (`recheck.model`)
    and DINOv3 (`instances.dino_model`) are loaded one after the other, each freed before
    the next, unless given (then their caches are not used). Returns meta.json (only
    `{"enabled": false}` when disabled)."""
    params = dict(cfg["detection"])
    if not params["enabled"]:
        print("[detect] disabled (detection.enabled = false)")
        return {"enabled": False}
    inst = dict(cfg["instances"])
    paths = OutputPaths(out_dir)
    folder = paths.detection
    folder.mkdir(parents=True, exist_ok=True)
    times: dict[str, float] = {}
    start = time.perf_counter()
    recon = load_recon(paths.recon / "recon.npz")
    grid = None
    if (paths.visibility / "grid.npz").is_file():
        from roomdiff.visibility import load_grid

        grid = load_grid(paths.visibility / "grid.npz")
    height, width = recon["depth"].shape[1:]
    times["load"] = time.perf_counter() - start

    t0 = time.perf_counter()
    prompts, vocab_doc = _vocabulary(paths, recon, cfg, params, vlm, folder)
    times["vocabulary"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw, det_meta = _detections(paths, recon, cfg, params, prompts, segmenter, folder)
    times["detections"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    min_px = int(params["min_mask_px"])
    detections: list[list[Detection]] = [[], []]
    det_log: list[dict[str, Any]] = []
    for r in range(len(RECORDINGS)):
        by_frame: dict[int, list[tuple[str, float, str]]] = {}
        for frame, phrase, score, counts in raw[r]:
            by_frame.setdefault(frame, []).append((phrase, score, counts))
        for frame in sorted(by_frame):  # one keyframe's full masks in memory at a time
            found = []
            for phrase, score, counts in by_frame[frame]:
                mask = decode_rle({"size": [height, width], "counts": counts})
                if keep_detection(mask, score, params["score_min"], min_px,
                                  params["max_mask_frac"]):
                    found.append((phrase, mask, score))
            if not found:
                continue
            world = unproject_depth(recon["depth"][frame], recon["K"][frame],
                                    recon["cam2world"][frame])
            for d in merge_frame_detections(found, params["merge_iou"], r, frame):
                d.points, d.colors = lift_detection(recon, d, params["lift_erode_px"],
                                                    inst["lift_voxel"], inst["lift_cell_m"],
                                                    inst["lift_link_m"], world)
                detections[r].append(d)
                det_log.append({"recording": RECORDINGS[r], "frame": frame,
                                "votes": {k: round(v, 3) for k, v in d.votes.items()},
                                "score": round(d.score, 3), "box": list(d.box),
                                "area": d.region.area, "points": len(d.points)})
            del found, world
    times["merge_and_lift"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    objects: list[list[DetObject]] = []
    weak: list[list[DetObject]] = []
    dropped: list[dict[str, Any]] = []
    for r in range(len(RECORDINGS)):
        objs, faint, gone = build_objects(detections[r], recon, params, inst["lift_voxel"])
        objects.append(objs)
        weak.append(faint)
        dropped += gone
    next_id = 1
    for objs in (*objects, *weak):  # weak objects last: ids after all objects
        for o in objs:
            o.id, next_id = next_id, next_id + 1
    times["cluster"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    feat_meta: dict[str, Any] = {}
    if any(objects) or any(weak):
        own = featurizer is None
        if own:
            from roomdiff.models import Dinov3Featurizer

            _free_gpu()
            featurizer = Dinov3Featurizer(inst["dino_model"], inst["device"])
        tokens: dict[int, np.ndarray] = {}

        def frame_tokens(frame: int) -> np.ndarray:
            if frame not in tokens:
                tokens[frame] = featurizer.patch_tokens(f"detect:{frame}", recon["images"][frame])
            return tokens[frame]

        for objs in (*objects, *weak):
            for o in objs:
                o.feature = object_feature(o, frame_tokens, inst["mask_erode_px"],
                                           params["feature_views"])
        feat_meta = {"model": getattr(featurizer, "model_id", type(featurizer).__name__),
                     "keyframes": len(tokens), "peak_vram_gb": _vram_peak()}
        del tokens
        if own:
            del featurizer
            _free_gpu(check=False)
    times["features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pairs = associate([*objects[0], *weak[0]], [*objects[1], *weak[1]], params)
    gated: list[dict[str, Any]] = []
    for r, objs in enumerate(objects):
        labels = Counter(o.top_label for o in objs)  # the (not weak) objects of the recording
        for o in objs:
            if o.decision == "unchanged":
                continue
            o.place = place_evidence(o.points, recon, 1 - r, grid)
            o.label_count = labels[o.top_label]
            o.gate = gate_reasons(o.place["observing_views"], o.mean_score, o.label_count,
                                  params)
            o.kind = None if o.gate else KINDS[r]
            if o.gate:
                gated.append(gate_record(o))
    times["associate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    records = [object_record(o, recon) for objs in (*objects, *weak) for o in objs]
    (folder / "clusters.json").write_text(json.dumps(
        {"image_size": [int(width), int(height)], "prompts": prompts, "clusters": records},
        indent=1))
    all_objs = [o for objs in (*objects, *weak) for o in objs]
    empty = [np.zeros((0, 3))]
    np.savez_compressed(
        folder / "clusters_points.npz",
        points=np.concatenate([o.points for o in all_objs] or empty).astype(np.float32),
        colors=np.concatenate([o.colors for o in all_objs] or empty).astype(np.uint8),
        cluster=np.concatenate([np.full(len(o.points), o.id) for o in all_objs]
                               or [np.zeros(0)]).astype(np.int32))
    mask_rows = [(o.id, v.frame, encode_rle(v.mask)["counts"]) for o in all_objs for v in o.views]
    np.savez_compressed(
        folder / "clusters_masks.npz",
        cluster=np.array([m[0] for m in mask_rows], np.int32),
        frame=np.array([m[1] for m in mask_rows], np.int32),
        rle=np.array([m[2] for m in mask_rows], dtype=np.str_),
        size=np.array([height, width], np.int64))
    if all_objs:
        np.save(folder / "features.npy", np.stack([o.feature for o in all_objs]).astype(np.float32))
    decisions = [{"id": o.id, "recording": RECORDINGS[o.rec], "label": o.top_label,
                  "decision": o.decision, "partner": o.partner, "kind": o.kind,
                  "observing_views": o.place.get("observing_views"), "gate": o.gate,
                  "nearest_compatible": o.nearest} for o in all_objs if not o.weak]
    (folder / "association.json").write_text(json.dumps(
        {"params": {k: params[k] for k in ("same_place_dist", "same_place_frac",
                                           "same_place_centre_min", "same_place_centre_rel",
                                           "label_cos_min", "same_feature_min",
                                           *GATE_PARAMS)},
         "decisions": decisions, "gate": gated, "pairs": pairs}, indent=1))
    (folder / "detections.json").write_text(json.dumps({"detections": det_log}))
    times["write"] = time.perf_counter() - t0
    times["total"] = time.perf_counter() - start

    def count(key: str, value: str | None) -> dict[str, int]:
        return {name: sum(getattr(o, key) == value for o in objects[r])
                for r, name in enumerate(RECORDINGS)}

    meta = {
        "enabled": True,
        "params": params,
        "models": {"vlm": vocab_doc["key"]["model"],
                   "sam3": getattr(segmenter, "model_id", None) or cfg["recheck"]["model"],
                   "dino": feat_meta.get("model")},
        "prompts": len(prompts),
        "vocabulary": vocab_doc["vlm"],
        "detections_raw": {name: len(raw[r]) for r, name in enumerate(RECORDINGS)},
        "detections": {name: len(detections[r]) for r, name in enumerate(RECORDINGS)},
        "sam3": det_meta,
        "features": feat_meta,
        "objects": {name: len(objects[r]) for r, name in enumerate(RECORDINGS)},
        "weak_objects": {name: len(weak[r]) for r, name in enumerate(RECORDINGS)},
        "objects_dropped": dropped,
        "decisions": {d: count("decision", d) for d in DECISIONS},
        "candidates": {k: count("kind", k) for k in KINDS},
        "gate": {"params": {k: params[k] for k in GATE_PARAMS},
                 "dropped": {name: sum(g["recording"] == name for g in gated)
                             for name in RECORDINGS},
                 "reasons": {k: sum(k in g["reasons"] for g in gated) for k in GATE_REASONS}},
        "runtime_s": {k: round(v, 2) for k, v in times.items()},
        "peak_ram_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    all_objs = [o for objs in objects for o in objs]
    cands = sum(o.kind is not None for o in all_objs)
    print(f"[detect] {len(prompts)} prompts, detections {meta['detections']['before']} / "
          f"{meta['detections']['after']} -> objects {meta['objects']['before']} / "
          f"{meta['objects']['after']} -> {cands} candidates "
          f"({sum(o.decision == 'unchanged' for o in all_objs)} unchanged, {len(gated)} "
          f"gated out), {times['total']:.1f} s -> {folder}")
    return meta
