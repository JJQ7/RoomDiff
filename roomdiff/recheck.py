"""Optional re-check of the M5 changes: label-conditioned re-detection with SAM 3.

After `roomdiff describe` every change has a `label`. SAM 3 segments all instances of a
noun phrase in an image, which gives a third source of evidence next to the geometry
(free space) and the VLM judge. For every kept `removed`, `added` and `moved` change
(VLM verdict confirmed or unclear; unverified changes are skipped):

1. Prompt: the label; if it names only a material or a vague thing ("grey fabric",
   "object"), the head noun of `object_description` ("A black jacket lying on ..." ->
   "jacket"); neither -> skipped.
2. Own side: the `max_views` best keyframes of the object's own recording (moved: both
   recordings) that hold a stored per-keyframe SAM 2.1 mask of it (`instances/<rec>_
   masks.npz`), ranked like the crops (view scorer x SAM score) with viewing directions at
   least `instances.min_view_angle` apart; without stored masks the projected instance
   points stand in. Found = a detection with score >= `score_min` whose mask overlaps the
   stored mask by IoU >= `overlap_min`. Nothing found = the label is unsupported
   (evidence only).
3. Other side: the place of the object in the other recording, in its `max_views` best
   views of it (the crop views: ranked by the view scorer, not occluded, >= 15 % of the
   points visible, directions apart). Place = the pixels where the instance points
   project and are not hidden behind a nearer surface, gaps between the points closed;
   moved changes minus the pixels of the object's other position (a view where less than
   `MIN_PLACE_LEFT` of the place remains is replaced by the next one). A detection is at
   the place if score >= `score_min`, IoU with the place >= `overlap_min` and at least
   `AT_PLACE_FRAC` of its pixels inside the place have an observed depth within
   `depth_tol` of the object's surface (an instance of the prompt behind or in front of
   the place is not at it). Removed: after views of the old place; added: before views
   of the new place; moved: after views of the old place and before views of the new
   place. Found at the place = a contradiction.
4. Verdict: skipped (no prompt or no view), contradicted (>= 1 contradicting view;
   moved: at both places, one place alone is often a sibling such as the other curtain of
   a window), unsupported (label not found on the object), else supported. Confidence x
   `unsupported_factor` without own-side support and x `contradicted_factor` when
   contradicted. A change moves to `rejected` only when contradicted in >= `reject_views`
   views; its `vlm_verdict` stays, `evidence.recheck_verdict` = "contradicted".

The first run backs the M5 document up to `recheck/changes_m5.json`; later runs start from
that backup (the factors are never applied twice; re-run `roomdiff describe` to refresh
it). Writes `<out>/changes.json` (same schema and ids, `evidence.recheck` on every kept
change), `recheck/log.jsonl` (one line per image and prompt), `recheck/meta.json` and
`report.md`. With `recheck.enabled: false` nothing is read or written.
"""

from __future__ import annotations

import gc
import json
import os
import re
import resource
import shutil
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from roomdiff.changes import CROP_DEPTH_TOL
from roomdiff.instances import decode_rle, load_masks, select_views
from roomdiff.models import TextSegmenter
from roomdiff.paths import RECORDINGS, OutputPaths
from roomdiff.projection import MIN_VISIBLE_FRAC, project_to_pixels, score_views, subsample
from roomdiff.recon import load_recon
from roomdiff.semantics import (
    ARTICLES,
    CUT_WORDS,
    head_noun,
    instance_points,
    sort_changes,
    summary_line,
)

VERDICTS = ("supported", "unsupported", "contradicted", "skipped")
CHECKED_TYPES = ("removed", "added", "moved")
DETECTION_FLOOR = 0.1  # SAM 3 detections below this score are not returned at all
AT_PLACE_FRAC = 0.5  # a detection is at the place if this share of its place pixels is
# depth-consistent with the object's surface
MIN_PLACE_LEFT = 0.25  # moved: share of a place view left outside the object's other position
MAX_REGION_POINTS = 100_000  # instance points projected per view (fixed subset)
MAX_RESIDENT_GIB = 4.0  # refuse to load SAM 3 while more than this is still on the GPU
# (an earlier stage of the same `roomdiff run` process, e.g. Qwen3-VL, was not freed)
PLACES = {"old": "after", "new": "before"}  # place -> recording that looks at it
# a label naming only a material or a vague thing (the VLM is asked not to, but may)
MATERIAL_WORDS = {"fabric", "cloth", "textile", "material", "leather", "wood", "wooden",
                  "plastic", "metal", "metallic", "paper", "cardboard", "glass", "foam",
                  "cotton", "wool", "linen", "rubber", "item", "thing", "stuff", "object"}
# words that end the first noun phrase of an object description
PHRASE_END = CUT_WORDS | {"of", "over", "that", "which", "is", "are", "was", "and",
                          "lying", "sitting", "standing", "hanging", "leaning", "draped",
                          "placed", "resting", "positioned", "located", "covering",
                          "mounted", "used"}


# ------------------------------------------------------------------ prompt


def description_noun(text: str) -> str | None:
    """Head noun of the first noun phrase of a description: the last word before the first
    preposition or participle ("A crumpled black jacket lying on the floor." -> "jacket")."""
    first = re.split(r"[.;:!?]", text.lower(), maxsplit=1)[0]
    phrase: list[str] = []
    for word in re.sub(r"[^a-z\- ]", " ", first).split():
        if not phrase and word in ARTICLES:
            continue
        if phrase and word in PHRASE_END:
            break
        phrase.append(word)
    return phrase[-1] if phrase else None


def prompt_for(change: Mapping[str, Any]) -> str | None:
    """SAM 3 text prompt of a change: its label, or the head noun of its
    `object_description` if the label names only a material or a vague thing; None if
    neither names an object."""
    label = str(change.get("label") or "").strip().lower()
    if label and not {label.split()[-1], head_noun(label)} & MATERIAL_WORDS:
        return label
    noun = description_noun(str(change.get("object_description") or ""))
    return noun if noun and noun not in MATERIAL_WORDS else None


# ------------------------------------------------------------------ views


@dataclass
class Check:
    """One SAM 3 query of a change: keyframe `frame` of `recording` and the pixels to
    compare the detections with (`region`; own: the stored mask of the object, place: the
    place as seen from that keyframe, with the object's surface depth `zsurf`)."""

    cid: int
    role: str  # "own" | "place"
    recording: str  # recording of the keyframe
    frame: int
    region: np.ndarray  # (H, W) bool at recon resolution
    place: str | None = None  # place checks: "old" (before position) | "new" (after)
    zsurf: np.ndarray | None = None  # place checks: (H, W) object depth, inf outside
    reference: str = "mask"  # own checks: "mask" (stored) | "points" (projected points)


def project_region(points: np.ndarray, frame: int, recon: Mapping[str, Any], mode: str,
                   spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (H, W) where the object `points` project in keyframe `frame`, the gaps
    between neighbouring points (`spacing` m apart) closed by a morphological closing (not
    a dilation: the outline stays where it is), and the nearest point depth there (inf
    elsewhere). mode "surface": only depth-consistent points (|observed - z| <=
    CROP_DEPTH_TOL, the object in its own recording); "place": points not hidden behind a
    nearer surface (observed >= z - CROP_DEPTH_TOL, the other recording at the place)."""
    height, width = recon["depth"].shape[1:]
    pts = subsample(np.asarray(points, np.float64).reshape(-1, 3), MAX_REGION_POINTS)
    zbuf = np.full((height, width), np.inf, np.float32)
    if not len(pts):
        return np.zeros((height, width), bool), zbuf
    col, row, z, inside = project_to_pixels(pts, recon["K"][frame], recon["cam2world"][frame],
                                            width, height)
    observed = np.zeros(len(pts))
    observed[inside] = recon["depth"][frame][row[inside], col[inside]]
    ok = inside.copy()
    ok[inside] &= recon["valid"][frame][row[inside], col[inside]]
    if mode == "surface":
        keep = ok & (np.abs(observed - z) <= CROP_DEPTH_TOL)
    else:
        keep = ok & (observed >= z - CROP_DEPTH_TOL)
    if not keep.any():
        return np.zeros((height, width), bool), zbuf
    np.minimum.at(zbuf, (row[keep], col[keep]), z[keep].astype(np.float32))
    thinning = np.sqrt(max(1.0, len(points) / MAX_REGION_POINTS))  # subsampled: wider gaps
    gap = float(recon["K"][frame][0, 0]) * spacing * thinning / max(float(np.median(z[keep])),
                                                                    1e-3)
    size = int(np.clip(2 * np.ceil(gap) + 1, 3, 15))
    padded = np.pad(np.isfinite(zbuf), size)  # no erosion from the image border
    region = ndimage.binary_closing(padded, np.ones((size, size), bool))[size:-size, size:-size]
    # every closed pixel lies within size // 2 of a point: its nearest depth is defined there
    zsurf = np.where(region, ndimage.minimum_filter(zbuf, size=size), np.inf)
    return region, zsurf


def pick_diverse(frames: Sequence[int], k: int, min_angle_deg: float, cam2world: np.ndarray,
                 centroid: np.ndarray) -> list[int]:
    """Up to `k` of the ranked `frames`: greedily those whose viewing directions (camera ->
    `centroid`) are at least `min_angle_deg` apart, remaining slots by rank (as
    `roomdiff.instances.select_views`)."""
    directions = {}
    for f in frames:
        d = cam2world[f][:3, 3] - centroid
        directions[f] = d / max(float(np.linalg.norm(d)), 1e-9)
    min_cos = np.cos(np.radians(min_angle_deg))
    chosen: list[int] = []
    for f in frames:
        if len(chosen) < k and all(directions[f] @ directions[g] < min_cos for g in chosen):
            chosen.append(f)
    for f in frames:
        if len(chosen) < k and f not in chosen:
            chosen.append(f)
    return chosen


def stored_masks(masks: Mapping[str, Sequence[tuple[int, float, Mapping[str, Any]]]],
                 file: str, ids: Sequence[int]) -> list[tuple[float, Mapping[str, Any]]]:
    """(score, RLE) of the stored per-keyframe masks of the instances `ids` in `file`."""
    return [(score, rle) for inst, score, rle in masks.get(file, []) if inst in ids]


def own_checks(cid: int, rec: str, ids: Sequence[int], points: np.ndarray,
               recon: Mapping[str, Any], masks: Mapping[str, Any], k: int, min_angle: float,
               spacing: float) -> list[Check]:
    """Own-side checks of one change side: the `k` best keyframes of recording `rec` with
    a stored mask of the instances `ids` (view score x SAM score, as the crops; directions
    apart); the stored masks (united over `ids`) are the reference. Without stored masks,
    the best views by the view scorer with the projected points as the reference."""
    frames = np.flatnonzero(recon["rec"] == RECORDINGS.index(rec))
    ranked = [v for v in score_views(points, frames, recon["cam2world"], recon["K"],
                                     recon["depth"], recon["valid"], CROP_DEPTH_TOL)
              if v.score > 0 and v.visible >= MIN_VISIBLE_FRAC]
    files = [str(f) for f in recon["files"]]
    sam = {v.frame: max((s for s, _ in stored_masks(masks, files[v.frame], ids)), default=None)
           for v in ranked}
    with_mask = sorted((v for v in ranked if sam[v.frame] is not None),
                       key=lambda v: (-v.score * sam[v.frame], v.frame))
    centroid = np.asarray(points, np.float64).mean(0)
    checks = []
    for f in pick_diverse([v.frame for v in with_mask], k, min_angle, recon["cam2world"],
                          centroid):
        mask = np.logical_or.reduce([decode_rle(rle) for _, rle in
                                     stored_masks(masks, files[f], ids)])
        checks.append(Check(cid, "own", rec, f, mask))
    if checks:
        return checks
    for f in pick_diverse([v.frame for v in ranked], k, min_angle, recon["cam2world"], centroid):
        region, _ = project_region(points, f, recon, "surface", spacing)
        if region.any():
            checks.append(Check(cid, "own", rec, f, region, reference="points"))
    return checks


def place_checks(cid: int, place: str, points: np.ndarray, recon: Mapping[str, Any], k: int,
                 min_angle: float, spacing: float, other: np.ndarray | None = None
                 ) -> list[Check]:
    """Checks of a place (`points`: the object there) in the recording that looks at it
    (`PLACES`): its best views as for the crops (not occluded, >= 15 % visible, directions
    apart). `other`: the object's points at its other position (moved), whose pixels are
    removed from the place; a view keeping less than `MIN_PLACE_LEFT` of it is skipped."""
    rec = PLACES[place]
    frames = np.flatnonzero(recon["rec"] == RECORDINGS.index(rec))
    checks = []
    for f, _ in select_views(points, frames, recon, 3 * k, CROP_DEPTH_TOL, min_angle,
                             on_surface=False):
        region, zsurf = project_region(points, f, recon, "place", spacing)
        if other is not None and region.any():
            there, _ = project_region(other, f, recon, "place", spacing)
            kept = region & ~ndimage.binary_dilation(there, iterations=2)
            if kept.sum() < MIN_PLACE_LEFT * region.sum():
                continue
            region = kept
        if region.any():
            checks.append(Check(cid, "place", rec, f, region, place, zsurf))
        if len(checks) == k:
            break
    return checks


def side_points(change: Mapping[str, Any], name: str,
                points: Mapping[str, Mapping[int, np.ndarray]]) -> tuple[list[int], np.ndarray]:
    """Instance ids and points (concatenated) of one side of a change."""
    side = change[name]
    ids = list(side.get("instance_ids") or [side["instance_id"]])
    own = [points.get(name, {}).get(i) for i in ids]
    own = [p for p in own if p is not None and len(p)]
    return ids, np.concatenate(own) if own else np.zeros((0, 3))


def plan_checks(change: Mapping[str, Any], recon: Mapping[str, Any],
                points: Mapping[str, Mapping[int, np.ndarray]],
                masks: Mapping[str, Mapping[str, Any]], k: int, min_angle: float,
                spacing: float) -> list[Check]:
    """All checks of one removed / added / moved change (module docstring, 2-3)."""
    cid, kind = change["id"], change["type"]
    sides = {name: side_points(change, name, points) for name in RECORDINGS if change.get(name)}
    checks: list[Check] = []
    for name, (ids, pts) in sides.items():
        if len(pts):
            checks += own_checks(cid, name, ids, pts, recon, masks[name], k, min_angle, spacing)
    before = sides.get("before", ([], np.zeros((0, 3))))[1]
    after = sides.get("after", ([], np.zeros((0, 3))))[1]
    moved = kind == "moved" and len(before) and len(after)
    if kind in ("removed", "moved") and len(before):
        checks += place_checks(cid, "old", before, recon, k, min_angle, spacing,
                               after if moved else None)
    if kind in ("added", "moved") and len(after):
        checks += place_checks(cid, "new", after, recon, k, min_angle, spacing,
                               before if moved else None)
    return checks


# ------------------------------------------------------------------ evidence


def match_detections(check: Check, masks: np.ndarray, scores: np.ndarray,
                     recon: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Best detection of one check: `score`, `overlap` (mask IoU with the check's region),
    for places `at_place` (share of its pixels inside the place, with valid depth, within
    `depth_tol` of the object's surface; None without such pixels), and `found` (score >=
    score_min, overlap >= overlap_min, at the place). Best = found first, then by IoU and
    score; None values without detections."""
    region = check.region
    n_region = int(region.sum())
    depth, valid = recon["depth"][check.frame], recon["valid"][check.frame]
    best: dict[str, Any] = {"score": None, "overlap": None, "found": False}
    if check.role == "place":
        best["at_place"] = None
    best_key: tuple[bool, float, float] | None = None
    for mask, score in zip(masks, scores):
        inter = mask & region
        n_inter = int(inter.sum())
        union = int(mask.sum()) + n_region - n_inter
        iou = n_inter / union if union else 0.0
        found = bool(score >= params["score_min"] and iou >= params["overlap_min"])
        record: dict[str, Any] = {"score": round(float(score), 3), "overlap": round(iou, 3)}
        if check.role == "place":
            sel = inter & valid & np.isfinite(check.zsurf)
            at = float((np.abs(depth[sel] - check.zsurf[sel]) <= params["depth_tol"]).mean()) \
                if sel.any() else None
            record["at_place"] = None if at is None else round(at, 3)
            found = found and at is not None and at >= AT_PLACE_FRAC
        key = (found, iou, float(score))
        if best_key is None or key > best_key:
            best, best_key = {**record, "found": found}, key
    return {"detections": len(scores), **best}


def own_found(views: Sequence[Mapping[str, Any]]) -> bool | None:
    """The label was found on the object in every recording that has own views (None
    without own views)."""
    if not views:
        return None
    recordings = {v["recording"] for v in views}
    return all(any(v["found"] for v in views if v["recording"] == r) for r in recordings)


def decide(kind: str, own: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]],
           params: Mapping[str, Any]) -> tuple[str, float, bool]:
    """(verdict, confidence factor, reject) from the own-side and place view records
    (module docstring, 4). A moved change is contradicted only if both of its places show
    the prompt (the object never left and was already there): one of them alone is often
    a sibling (the other curtain of the window, a second chair)."""
    found = own_found(own)
    contra = [v for v in other if v["found"]]
    contradicted = bool(contra) and (kind != "moved" or len({v["place"] for v in contra}) == 2)
    reject = contradicted and len(contra) >= params["reject_views"]
    if not own and not other:
        verdict = "skipped"
    elif contradicted:
        verdict = "contradicted"
    elif found is False:
        verdict = "unsupported"
    else:
        verdict = "supported"
    factor = (params["unsupported_factor"] if found is False else 1.0) * (
        params["contradicted_factor"] if contradicted else 1.0)
    return verdict, round(factor, 4), reject


def _views(n: int, recording: str) -> str:
    return f"{n} {recording} view{'s' if n != 1 else ''}"


def reason_text(prompt: str, own: Sequence[Mapping[str, Any]],
                other: Sequence[Mapping[str, Any]]) -> str:
    """"'black jacket' found on the object in 2 of 2 own views (0.93), absent from the old
    place in 2 after views"."""
    if not own and not other:
        return "no keyframe shows the object or its place"
    parts = []
    if own:
        hits = [v for v in own if v["found"]]
        recordings = sorted({v["recording"] for v in own}, key=RECORDINGS.index)
        if hits:
            parts.append(f"'{prompt}' found on the object in {len(hits)} of {len(own)} own "
                         f"views ({max(v['score'] for v in hits):.2f})")
        else:
            parts.append(f"'{prompt}' not found on the object in "
                         f"{_views(len(own), '/'.join(recordings))}")
    for place, recording in PLACES.items():
        views = [v for v in other if v["place"] == place]
        hits = [v for v in views if v["found"]]
        if hits:
            top = max(hits, key=lambda v: (v["overlap"], v["score"]))
            parts.append(f"found at the {place} place in {len(hits)} of "
                         f"{_views(len(views), recording)} (score {top['score']:.2f}, "
                         f"IoU {top['overlap']:.2f})")
        elif views:
            parts.append(f"absent from the {place} place in {_views(len(views), recording)}")
    return ", ".join(parts)


def best_of(views: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The most telling view record: found first, then by overlap and score."""
    scored = [v for v in views if v.get("score") is not None]
    if not scored:
        return {}
    return max(scored, key=lambda v: (v["found"], v["overlap"], v["score"]))


def recheck_evidence(change: Mapping[str, Any], prompt: str | None,
                     checks: Sequence[Check], results: Mapping[int, Mapping[str, Any]],
                     recon: Mapping[str, Any], params: Mapping[str, Any]
                     ) -> tuple[dict[str, Any], bool]:
    """`evidence.recheck` of one change from the results of its checks (index -> match
    record), and whether to reject it."""
    if change["type"] not in CHECKED_TYPES:
        return skipped(prompt, f"{change['type']} changes are not re-checked"), False
    if prompt is None:
        return skipped(None, "the label names no object"), False
    files = [str(f) for f in recon["files"]]
    own, other = [], []
    for n, check in enumerate(checks):
        record = {"recording": check.recording, "frame": check.frame,
                  "file": files[check.frame], **results[n]}
        if check.role == "own":
            if check.reference != "mask":
                record["reference"] = check.reference
            own.append(record)
        else:
            other.append({"place": check.place, **record})
    verdict, factor, reject = decide(change["type"], own, other, params)
    top_own, top_other = best_of(own), best_of(other)
    evidence = {
        "prompt": prompt,
        "own": {"views": own, "found": own_found(own), "score": top_own.get("score"),
                "overlap": top_own.get("overlap")},
        "other": {"views": other, "found": any(v["found"] for v in other) if other else None,
                  "found_views": sum(v["found"] for v in other),
                  "score": top_other.get("score"), "overlap": top_other.get("overlap")},
        "verdict": verdict, "reason": reason_text(prompt, own, other), "factor": factor,
    }
    return evidence, reject


def skipped(prompt: str | None, reason: str) -> dict[str, Any]:
    return {"prompt": prompt, "own": None, "other": None, "verdict": "skipped",
            "reason": reason, "factor": 1.0}


# ------------------------------------------------------------------ stage


def is_rechecked(doc: Mapping[str, Any]) -> bool:
    return any("recheck" in (c.get("evidence") or {})
               for c in [*doc.get("changes", []), *doc.get("rejected", [])])


def load_m5_changes(paths: OutputPaths, folder: Path) -> dict[str, Any]:
    """The M5 changes document: `changes.json` if the re-check has not processed it yet
    (it is then backed up to `recheck/changes_m5.json`), else the backup."""
    doc = json.loads(paths.changes_json.read_text())
    if "rejected" not in doc:
        raise ValueError(f"{paths.changes_json} has not been through `roomdiff describe` "
                         "yet: the re-check needs its labels")
    backup = folder / "changes_m5.json"
    if not is_rechecked(doc):
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.changes_json, backup)
        return doc
    if not backup.is_file():
        raise FileNotFoundError(f"{paths.changes_json} was already re-checked and {backup} "
                                "is missing; re-run `roomdiff describe` first")
    return json.loads(backup.read_text())


def updated_summary(summary: str, kept: Sequence[Mapping[str, Any]],
                    dropped: Sequence[Mapping[str, Any]]) -> str:
    """The M5 summary with the counts of `kept` and a note on the changes the re-check
    rejected (`dropped`); unchanged without such changes."""
    if not dropped:
        return summary
    rest = re.sub(r"^\d+ changes?: \d+ moved, \d+ added, \d+ removed; \d+ unverified\.\s*", "",
                  summary)
    names = ", ".join(f"change {c['id']} ({c['type']} {c['label']})" for c in dropped)
    note = (f"A SAM 3 re-check later rejected {names}: the object was still found at its "
            "place.")
    return " ".join(x for x in (f"{summary_line(kept)}.", rest, note) if x)


def keyframe_image(paths: OutputPaths, recon: Mapping[str, Any], frame: int
                   ) -> tuple[np.ndarray, str]:
    """Full-resolution keyframe (recon image as a fallback) and its path below `<out>`."""
    from PIL import Image

    rec = RECORDINGS[int(recon["rec"][frame])]
    file = paths.keyframes_of(rec) / str(recon["files"][frame])
    if file.is_file():
        return np.asarray(Image.open(file).convert("RGB")), str(file.relative_to(paths.root))
    return recon["images"][frame], f"recon/recon.npz:images[{frame}]"


def _release_gpu() -> None:
    """Free what earlier stages of this process left on the GPU and reset the peak counter;
    refuse to go on if more than MAX_RESIDENT_GIB is still allocated."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated() / 2**30
    if resident > MAX_RESIDENT_GIB:
        raise RuntimeError(f"{resident:.1f} GiB still allocated on the GPU: refusing to load "
                           "SAM 3 next to another model; run `roomdiff recheck` on its own")
    torch.cuda.reset_peak_memory_stats()


def _peak_memory() -> dict[str, Any]:
    torch = sys.modules.get("torch")
    vram = None
    if torch is not None and torch.cuda.is_initialized():
        vram = {"allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    return {"peak_vram_gb": vram,
            "peak_ram_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}


def _write_atomic(path: Path, text: str) -> None:
    """Replace `path` by a new file (never truncates it in place: a hard-linked copy of
    an output folder keeps its original)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def run_recheck(out_dir: str | Path, cfg: Mapping[str, Any],
                segmenter: TextSegmenter | None = None) -> dict[str, Any] | None:
    """Re-check the changes of `<out>/changes.json` with SAM 3 (module docstring) and
    rewrite it, `recheck/` and `report.md`. Loads SAM 3 unless `segmenter` is given.
    Returns the new changes document, None if `recheck.enabled` is false."""
    params = dict(cfg["recheck"])
    if not params["enabled"]:
        print("[recheck] disabled (recheck.enabled: false): changes.json left as it is")
        return None
    from roomdiff.report import write_report

    paths = OutputPaths(out_dir)
    folder = paths.root / "recheck"
    times: dict[str, float] = {}
    start = time.perf_counter()
    doc = load_m5_changes(paths, folder)
    recon = load_recon(paths.recon / "recon.npz")
    points = instance_points(paths)
    masks = {name: load_masks(paths.instances / f"{name}_masks.npz")
             if (paths.instances / f"{name}_masks.npz").is_file() else {} for name in RECORDINGS}
    times["load"] = time.perf_counter() - start

    t0 = time.perf_counter()
    k, min_angle = int(params["max_views"]), float(cfg["instances"]["min_view_angle"])
    spacing = float(cfg["instances"]["lift_voxel"])
    plans: dict[int, tuple[str | None, list[Check]]] = {}
    for c in doc["changes"]:
        prompt = prompt_for(c) if c["type"] in CHECKED_TYPES else None
        checks = plan_checks(c, recon, points, masks, k, min_angle, spacing) \
            if prompt is not None else []
        plans[c["id"]] = (prompt, checks)
    queries: dict[int, set[str]] = {}
    for prompt, checks in plans.values():
        for check in checks:
            queries.setdefault(check.frame, set()).add(prompt)
    times["plan"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    own_model = segmenter is None and bool(queries)
    if own_model:
        from roomdiff.models import Sam3TextSegmenter

        _release_gpu()
        segmenter = Sam3TextSegmenter(params["model"], params["device"], DETECTION_FLOOR)
    times["load_model"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    height, width = recon["depth"].shape[1:]
    detections: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    image_seconds: list[float] = []
    folder.mkdir(parents=True, exist_ok=True)
    tags: dict[tuple[int, str], list[str]] = {}
    for prompt, checks in plans.values():
        for check in checks:
            tag = f"{check.role}:{check.cid}:{check.place or check.recording}"
            tags.setdefault((check.frame, prompt), []).append(tag)
    with open(folder / "log.jsonl", "w", encoding="utf-8") as log:
        for frame in sorted(queries):
            image, name = keyframe_image(paths, recon, frame)
            prompts = sorted(queries[frame])
            t1 = time.perf_counter()
            found = segmenter.detect(name, image, prompts, size=(height, width))
            seconds = time.perf_counter() - t1
            image_seconds.append(seconds)
            for prompt in prompts:
                masks_k, scores_k = found[prompt]
                detections[(frame, prompt)] = (masks_k, scores_k)
                answer = []
                for m, s in zip(masks_k, scores_k):
                    rows, cols = np.nonzero(m)
                    box = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())] \
                        if len(rows) else None
                    answer.append({"score": round(float(s), 3), "box": box, "area": len(rows)})
                log.write(json.dumps({"tag": ",".join(tags[(frame, prompt)]), "images": [name],
                                      "prompt": prompt, "answer": answer,
                                      "seconds": round(seconds / len(prompts), 3)}) + "\n")
    times["inference"] = time.perf_counter() - t0
    model_info = {"checkpoint": str(getattr(segmenter, "checkpoint", "")) or None,
                  "missing_keys": getattr(segmenter, "missing_keys", None)}
    peak = _peak_memory()
    if own_model:  # free SAM 3 before anything else runs in this process
        del segmenter
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    t0 = time.perf_counter()
    kept, dropped = [], []
    confidence: dict[str, list[float]] = {}
    for c in doc["changes"]:
        prompt, checks = plans[c["id"]]
        results = {n: match_detections(ch, *detections[(ch.frame, prompt)], recon, params)
                   for n, ch in enumerate(checks)}
        evidence, reject = recheck_evidence(c, prompt, checks, results, recon, params)
        old = c["confidence"]
        c["confidence"] = round(old * evidence["factor"], 3)
        confidence[str(c["id"])] = [old, c["confidence"]]
        c["evidence"]["recheck"] = evidence
        if reject:
            c["evidence"]["recheck_verdict"] = "contradicted"
            dropped.append(c)
        else:
            kept.append(c)
    result = {"scene": doc["scene"], "changes": sort_changes(kept),
              "rejected": sort_changes([*doc["rejected"], *dropped]),
              "summary": updated_summary(doc["summary"], kept, dropped)}
    if "relations_info" in doc:  # texts drop relations to the newly rejected (change_text)
        result["relations_info"] = doc["relations_info"]
    _write_atomic(paths.changes_json, json.dumps(result, indent=2))
    times["decide"] = time.perf_counter() - t0
    times["total"] = time.perf_counter() - start

    verdicts = {str(c["id"]): c["evidence"]["recheck"]["verdict"] for c in doc["changes"]}
    counts = Counter(verdicts.values())
    meta = {
        "model": params["model"], **model_info,
        "params": params,
        "runtime_s": {key: round(v, 2) for key, v in times.items()},
        "images": len(image_seconds),
        "queries": sum(len(p) for p in queries.values()),
        "seconds_per_image": None if not image_seconds else {
            "mean": round(float(np.mean(image_seconds)), 3),
            "median": round(float(np.median(image_seconds)), 3),
            "max": round(float(np.max(image_seconds)), 3)},
        "counts": {**{v: counts[v] for v in VERDICTS}, "rejected": len(dropped)},
        "verdicts": verdicts,
        "confidence": confidence,
        "rejected": [{"id": c["id"], "type": c["type"], "label": c["label"],
                      "reason": c["evidence"]["recheck"]["reason"]} for c in dropped],
        **peak,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[recheck] {dict(meta['counts'])}, {meta['images']} images, "
          f"{meta['queries']} prompts, {times['total']:.1f} s -> {paths.changes_json}")
    write_report(paths.root)
    return result
