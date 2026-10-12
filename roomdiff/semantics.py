"""M5 semantics: name, verify, re-pair and describe the M4 changes with a local VLM.

Steps on `<out>/changes.json` (the M4 version is kept in `semantics/changes_m4.json`, so
the stage can be re-run):

1. Naming: the owner crop(s) of every change (moved: before and after) -> `label`
   (normalised: lowercase, no articles, at most 3 words), `attributes`,
   `object_description`.
2. Fragments: two `removed` (or two `added`) changes with matching labels whose oriented
   boxes are within `fragment_gap_m` of each other become one change if the VLM says both
   owner crops show parts of one physical object (a chair seat and its base): geometry from
   the union of their instance points, confidence = max, the larger fragment's id, crops
   and evidence (+ `<side>_part` crop, summed free rays, `merged_from`).
3. Re-pairing: a `removed` change and an `added` / after-only `unverified` change with a
   matching label (same label or head noun) whose centroids are closer than
   `repair_max_distance` become one `moved` change if the VLM says both crops show the
   same physical object. A `moved` change absorbs an `added` change with a matching label
   whose box lies within `merge_max_distance` of the moved after-box (gap between the
   boxes; a spread-out quilt is one object) if the VLM says both are parts of one object.
4. Judge: type-specific paired-crop questions -> `evidence.vlm_verdict` ("confirmed" /
   "rejected" / "unclear") and `evidence.vlm_reason`. A candidate in-place replacement (a
   removed change `replaced_by` an added one at the same place, from the appearance
   channel) is judged once on both owner crops: "same" (lighting / viewpoint only) rejects
   both, "different" keeps both, "removed" / "added" keeps only that one, "unclear" keeps
   both as unclear; a single appearance-channel removed / added change gets the same
   question with the other recording's view of the place (same or the other type:
   rejected; different or its own type: confirmed), and one named like a bare surface or
   part of the room is rejected; moved also `motion_kind` ("rigid" /
   "deformed" / "unknown"). Rejected changes move to the `rejected` list; unclear ones keep
   `confidence * unclear_factor`. A rejected `moved` change keeps its geometry: it is split
   back into a `removed` (before side) and an `added` change (after side), each named
   again and judged with the removed / added question (object + the other recording's
   view of the place): confirmed halves keep the moved confidence, unclear ones get
   `confidence * split_factor`, rejected ones (e.g. a wall region) move to `rejected`;
   the kept halves then go through fragment merging (step 2) and re-pairing (step 3)
   once more, with the other changes only.
5. Descriptions: a template sentence from label, type, geometry (distance, size, place in
   the room, height above the floor = lowest 5 % of all valid points) and evidence; one
   text-only VLM call polishes all of them, a sentence is kept only if every number and
   the label survived.
6. Summary: "N changes: a moved, b added, c removed; d unverified" + a VLM paragraph.

Writes `<out>/changes.json`, `<out>/semantics/meta.json` and `<out>/semantics/log.jsonl`
(every prompt and answer). Prompts are module constants.
"""

from __future__ import annotations

import json
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
from PIL import Image

from roomdiff.changes import CHANGE_TYPES, instance_evidence
from roomdiff.geometry import floor_obb, obb_corners, obb_gap
from roomdiff.paths import RECORDINGS, OutputPaths
from roomdiff.vlm import BaseVLM

VERDICTS = ("confirmed", "rejected", "unclear")
MOTION_KINDS = ("rigid", "deformed", "unknown")
ARTICLES = {"a", "an", "the", "some"}
TRAILING_WORDS = {"of", "on", "in", "with", "and", "against", "at", "for", "to"}
CUT_WORDS = {"on", "in", "with", "against", "at", "near", "under", "behind", "from"}
# head nouns of labels that name the room itself: never re-paired, and an unverified
# region named like this cannot be confirmed as an object
BACKGROUND_NOUNS = {"wall", "floor", "ceiling", "door", "doorframe", "doorway", "window",
                    "mirror", "reflection", "shadow", "lighting", "corner", "room",
                    "background", "baseboard", "skirting", "radiator"}
BACKGROUND_LABELS = ("door frame", "window frame", "window sill", "windowsill")
# label similarity beyond the head noun: words naming the same kind of household object
SYNONYMS = {
    "bedding": {"quilt", "duvet", "comforter", "blanket", "bedspread", "bedding", "throw"},
    "clothing": {"jacket", "coat", "hoodie", "sweater", "jumper", "shirt", "pajama",
                 "pyjama", "clothe", "clothing", "garment", "robe", "trouser", "pant"},
    "curtain": {"curtain", "drape", "drapery"},
    "bag": {"bag", "backpack", "rucksack", "handbag", "purse", "tote", "satchel"},
    "chair": {"chair", "armchair", "stool", "seat"},
    "box": {"box", "carton", "package", "parcel"},
    "pillow": {"pillow", "cushion"},
}
FLOOR_PERCENTILE = 5.0  # floor = median height of the lowest 5 % of all valid points
OBJECT_PERCENTILE = 5.0  # object bottom = 5th percentile of its point heights
ON_FLOOR_M = 0.15  # an object whose bottom is lower than this is "on the floor"
POINT_STRIDE = 4  # every 4th pixel (rows and columns) for the floor estimate
NUMBER = re.compile(r"\d+(?:\.\d+)?")

ROOM_PARTS = ("wall, floor, ceiling, door, door frame, window, radiator, mirror "
              "reflection, shadow or lighting")
VERDICT_JSON = ('Answer with JSON only, reason first: {"reason": "one short sentence", '
                '"verdict": "confirmed" | "rejected" | "unclear"}')

# ------------------------------------------------------------------ prompts

OUTLINE = ("The thin coloured contour traces the exact region; the rectangle only frames it. "
           "Judge only what lies inside the contour, not other objects that merely fall "
           "inside the rectangle (if there is no contour, judge what the rectangle mainly "
           "contains).")
NAME_RULE = ("Name the kind of object as a person would call it (what it is or is used "
             "for), not just its material or a vague word such as 'fabric', 'cloth', 'item' "
             "or 'thing'; look at its shape, e.g. sleeves, collar, handles or legs.")

NAME_ONE = (
    "This is a photo of a room. A coloured outline and box mark one region. What is the "
    "main object inside the outline? If the outline covers only part of the room itself "
    f"({ROOM_PARTS}), name that instead.\n{OUTLINE}\n{NAME_RULE}\n"
    'Answer with JSON only: {"label": "short lowercase noun phrase of 1-3 words, e.g. '
    'office chair", "attributes": "colour and material in a few words", "description": '
    '"one sentence describing the object"}')
NAME_TWO = (
    "Image 1 and image 2 are photos of the same room taken at different times. In each, a "
    "coloured outline and box mark one object, which was moved or rearranged between the "
    "two photos. What is this object? If the outlines cover only part of the room itself "
    f"({ROOM_PARTS}), name that instead.\n{OUTLINE}\n{NAME_RULE}\n"
    'Answer with JSON only: {"label": "short lowercase noun phrase of 1-3 words, e.g. '
    'office chair", "attributes": "colour and material in a few words", "description": '
    '"one sentence describing the object"}')
JUDGE_REMOVED = (
    "Image 1 is from an earlier video of a room; a coloured outline and box highlight one "
    "object (probably a {label}). Image 2 is from a later video of the same room and looks "
    "at the same place (its box marks the same spot in 3D; the viewpoint differs). Is the "
    f"highlighted object absent from that place in image 2?\n{OUTLINE}\n"
    'Answer "confirmed" if it is no longer at that place, also when it is still somewhere '
    "in the room but was moved away, opened, closed or rearranged (for example a curtain "
    'pulled aside); "rejected" if it is still at the same place (possibly seen from another '
    "angle or partly hidden) or if the highlighted region in image 1 is not an object but "
    f"part of the room ({ROOM_PARTS}); \"unclear\" if image 2 does not show the place "
    f"well enough to tell.\n{VERDICT_JSON}")
JUDGE_ADDED = (
    "Image 1 is from an earlier video of a room and looks at one place (marked by a box). "
    "Image 2 is from a later video of the same room; a coloured outline and box highlight "
    "one object (probably a {label}) at that place (the viewpoint differs). Was the "
    "highlighted object absent from that place in image 1, i.e. was it added between the "
    f"two videos?\n{OUTLINE}\n"
    'Answer "confirmed" if image 1 shows the place without the object, also when the object '
    "was elsewhere in the room before or had a different shape there (for example a quilt "
    'that was bunched up); "rejected" if it was already at the same place in image 1 or if '
    f"the highlighted region in image 2 is not an object but part of the room ({ROOM_PARTS});"
    ' "unclear" if image 1 does not show the place well enough to tell.\n'
    f"{VERDICT_JSON}")
JUDGE_SAME = (
    "Image 1 (earlier video) and image 2 (later video) show the same room. Each has a "
    "coloured outline and box around one object (probably a {label}), at different places "
    "or in different poses. Are they the same physical object, seen at two different places "
    f"or in two different poses?\n{OUTLINE}\n"
    "Compare the two outlined objects carefully: colour, pattern, material, parts (legs, "
    "wheels, armrests, handles) and size. Two rigid objects of the same kind but with a "
    "different design (e.g. a different backrest or legs) are different objects. Soft "
    "objects (curtains, quilts, blankets, clothes) can change shape completely when opened, "
    "spread out or crumpled, and the lighting can differ between the videos: judge them by "
    "pattern, colour and material, not by shape or pose.\n"
    'Answer "confirmed" if they are the same object; "rejected" if they are different '
    "objects or if either highlighted region is not an object but part of the room "
    f"({ROOM_PARTS}); \"unclear\" if you cannot tell.\n"
    'Answer with JSON only, in this order: {"differences": "visible differences between the '
    'two outlined objects, or none", "reason": "one short sentence", "verdict": "confirmed" '
    '| "rejected" | "unclear"}')
# M4 moved pairs were already matched by 3D geometry and DINOv3 features: the judge only
# has to catch gross errors (re-pairing uses the strict JUDGE_SAME, it has no such prior)
JUDGE_MOVED = JUDGE_SAME.replace(
    "Answer \"confirmed\" if they are the same object;",
    "The two regions were already matched by 3D geometry and visual features, so answer "
    "\"rejected\" only if they are clearly different objects (a different kind of object, or "
    "a clearly different design of a rigid object) or part of the room; with partial views, "
    "soft objects or changed lighting answer \"unclear\" rather than \"rejected\" when "
    "unsure.\nAnswer \"confirmed\" if they are the same object;")
JUDGE_MOTION = (
    "Image 1 (earlier video) and image 2 (later video) highlight the same object (probably "
    "a {label}) with a coloured outline and box. Was it moved as a whole, keeping its shape "
    "(like a chair, box or ladder carried elsewhere), or was it only rearranged or deformed "
    "(a curtain opened or closed, a quilt spread out or crumpled, clothes rearranged)?\n"
    'Answer with JSON only, reason first: {"reason": "one short sentence", "motion": '
    '"rigid" | "deformed" | "unknown"}')
JUDGE_UNVERIFIED = (
    "This photo of a room has a coloured outline and box around one region. Is the "
    "highlighted region a distinct movable object (furniture, an item, a bag, clothes, a "
    "box, a bottle, ...) rather than part of the room itself (wall, floor, ceiling, door, "
    "door frame, window, radiator), a mirror reflection, a shadow or a lighting effect?\n"
    f"{OUTLINE}\nFirst check whether the outline lies inside a mirror: a framed reflective "
    "surface that shows another view of the room. Everything inside a mirror is a "
    "reflection, even if it looks like a bed or furniture.\n"
    'Answer "confirmed" if the outline mainly covers a distinct real object; "rejected" if '
    "it mainly covers part of the room, a reflection, a shadow or lighting; \"unclear\" if "
    "you cannot tell.\n"
    'Answer with JSON only, in this order: {"inside_mirror": true | false, "reason": "what '
    'the contour covers, in one short sentence", "part_of_room": true | false, "verdict": '
    '"confirmed" | "rejected" | "unclear"}')
JUDGE_PARTS = (
    "Image 1 and image 2 come from the same video of a room. Each has a coloured outline "
    "and box around a region (probably a {label}). Do both outlines show the same physical "
    "object (for example two parts of one quilt or one curtain), rather than two different "
    f"objects?\n{OUTLINE}\n"
    'Answer "confirmed" if it is the same object; "rejected" if they are different '
    f'objects; "unclear" if you cannot tell.\n{VERDICT_JSON}')
JUDGE_FRAGMENTS = (
    "Image 1 and image 2 come from the same video of a room. Each has a coloured outline "
    "and box around a region (probably a {label}); in 3D the two regions touch or are only "
    "a few centimetres apart. Are they two parts of one physical object (for example the "
    "seat and the base of one chair, or the top and the legs of one table), rather than two "
    "separate objects close together (for example two chairs side by side, or the two "
    "curtains of one window)?\n"
    f"{OUTLINE}\n"
    'Answer "confirmed" if they are parts of one object; "rejected" if they are separate '
    'objects; "unclear" if you cannot tell.\n'
    'Answer with JSON only, reason first: {"reason": "one short sentence", "verdict": '
    '"confirmed" | "rejected" | "unclear", "label": "short lowercase noun phrase of 1-3 '
    'words naming the whole object, e.g. office chair"}')
JUDGE_REPLACED = (
    "Image 1 is from an earlier video of a room and image 2 from a later video of the same "
    "room; both look at the same place (the viewpoints differ). A coloured box marks that "
    "place in each image ({labels}). The two videos may have been filmed under different "
    "lighting. What changed at this place between the two videos?\n"
    f"{OUTLINE}\n"
    "A bare surface (a bed sheet, mattress, table top, shelf, floor, rug or wall with "
    "nothing on it) counts as an empty place, not as an object.\n"
    'Answer "same" if both images show the same thing at that place (the same object, or '
    "the same empty surface) and only the lighting, brightness, exposure, shadows, "
    "viewpoint or a slight shift or fold differ; \"different\" if the object of image 1 was "
    "replaced by a different object; \"removed\" if image 1 shows an object there and image "
    "2 only the empty place; \"added\" if image 1 shows only the empty place and image 2 an "
    'object there; "unclear" if you cannot tell.\n'
    'Answer with JSON only, reason first: {"reason": "one short sentence", "change": '
    '"same" | "different" | "removed" | "added" | "unclear"}')
# answer of JUDGE_REPLACED -> (verdict of the removed change, verdict of the added change)
REPLACE_VERDICTS = {
    "same": ("rejected", "rejected"),
    "different": ("confirmed", "confirmed"),
    "removed": ("confirmed", "rejected"),
    "added": ("rejected", "confirmed"),
    "unclear": ("unclear", "unclear"),
}
# head nouns naming a bare surface: an appearance-channel change named like this is the
# place where something lies or lay (a bed sheet under a hairbrush), not an object
SURFACE_NOUNS = {"sheet", "bedsheet", "mattress", "tabletop", "countertop", "worktop",
                 "surface", "floorboard", "tile", "plank"}
POLISH = (
    "Rewrite each numbered sentence below in plain, fluent English for a non-expert reader; "
    "rephrase them, do not copy them word for word. Keep every number exactly as written "
    "(same digits and units), keep the object names and keep the meaning: do not add facts "
    "or interpretations such as how or why an object moved (no 'slid', 'fell' or 'same "
    "spot'). Call the two recordings 'the earlier video' and 'the later video'.\n"
    "Example input: The red chair (plastic; about 0.9 x 0.6 x 0.6 m) moved 1.2 m, from the "
    "far left part of the room to the right side of the room (on the floor both times); the "
    "after recording saw the old place empty from 5 viewpoints and the before recording saw "
    "the new place empty from 3 viewpoints.\n"
    "Example output: The red plastic chair (about 0.9 x 0.6 x 0.6 m) was moved 1.2 m across "
    "the floor, from the far left of the room to its right side; the later video shows its "
    "old spot empty from 5 viewpoints and the earlier video shows the new spot empty from 3 "
    "viewpoints.\n\n{items}\n\n"
    'Answer with JSON only: {{"sentences": ["...", "..."]}} with exactly {n} sentences in '
    "the same order.")
SUMMARY = (
    "Below is the list of changes detected between two videos of the same room (an earlier "
    "'before' video and a later 'after' video). Write one short paragraph (2-4 sentences) "
    "in plain language that describes how the room changed. Mention unverified items only "
    "as possible changes that could not be checked. Do not invent changes that are not "
    "listed and do not speculate about how or why objects moved.\n\n{items}\n\n"
    'Answer with JSON only: {{"summary": "..."}}')


# ------------------------------------------------------------------ labels


def normalise_label(text: Any) -> str:
    """Lowercase noun phrase without articles or punctuation, at most 3 words."""
    words = re.sub(r"[^a-z0-9\- ]", " ", str(text or "").lower()).split()
    while words and words[0] in ARTICLES:
        words.pop(0)
    cut = next((k for k, w in enumerate(words) if w in CUT_WORDS and k > 0), len(words))
    words = words[:cut][:3]  # "bed with grey sheets" -> "bed"
    while words and words[-1] in TRAILING_WORDS:
        words.pop()
    return " ".join(words) or "object"


def head_noun(label: str) -> str:
    """Last word of a label, naively singularised ("curtains" -> "curtain")."""
    word = label.split()[-1] if label.split() else ""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def label_groups(label: str) -> set[str]:
    """Synonym groups (SYNONYMS) of the words of a label."""
    words = {head_noun(w) for w in label.split()}
    return {group for group, members in SYNONYMS.items() if words & members}


def is_background(label: str) -> bool:
    """The label names part of the room ("wall", "door frame", "mirror reflection")."""
    return head_noun(label) in BACKGROUND_NOUNS or label.endswith(BACKGROUND_LABELS)


def labels_match(a: str, b: str) -> bool:
    """Same label, same head noun or a shared synonym group ("duvet cover" ~ "quilt"),
    and not a part of the room (wall, floor, ...)."""
    if "object" in (a, b) or is_background(a) or is_background(b):
        return False
    return a == b or head_noun(a) == head_noun(b) or bool(label_groups(a) & label_groups(b))


# ------------------------------------------------------------------ scene geometry


@dataclass
class SceneFrame:
    """Gravity-aligned frame of the room: `up`, horizontal `right` / `forward` (mean camera
    view direction), floor height and the horizontal extent (2nd-98th percentile)."""

    up: np.ndarray
    right: np.ndarray
    forward: np.ndarray
    floor: float
    lo: np.ndarray  # (2,) min along right, forward
    hi: np.ndarray  # (2,) max along right, forward

    def height(self, points: np.ndarray) -> float:
        """Height of the bottom of `points` (5th percentile) above the floor, >= 0."""
        h = np.asarray(points, np.float64) @ self.up
        return max(0.0, float(np.percentile(h, OBJECT_PERCENTILE)) - self.floor)

    def place(self, centroid: Sequence[float]) -> str:
        """Coarse place in the room as seen from the cameras, e.g. "in the far left part
        of the room"."""
        c = np.asarray(centroid, np.float64)
        pos = (np.array([c @ self.right, c @ self.forward]) - self.lo) / np.maximum(
            self.hi - self.lo, 1e-6)
        side, depth = (int(np.clip(np.floor(3 * p), 0, 2)) for p in pos)
        side_word = ("left", "centre", "right")[side]
        depth_word = ("near", "middle", "far")[depth]
        if depth == 1 and side == 1:
            return "in the middle of the room"
        if depth == 1:
            return f"on the {side_word} side of the room"
        if side == 1:
            return f"in the {depth_word} part of the room"
        return f"in the {depth_word} {side_word} part of the room"


def scene_frame(paths: OutputPaths) -> SceneFrame | None:
    """SceneFrame from `recon/recon.npz` (None if there is no reconstruction)."""
    file = paths.recon / "recon.npz"
    if not file.is_file():
        return None
    from roomdiff.geometry import estimate_up, unproject_depth
    from roomdiff.recon import is_levelled, load_recon
    from roomdiff.viewer_export import horizontal_view_dir

    recon = load_recon(file)
    s = POINT_STRIDE
    points = [unproject_depth(recon["depth"][i], recon["K"][i], recon["cam2world"][i])[::s, ::s][
        recon["valid"][i][::s, ::s]] for i in range(len(recon["rec"]))]
    pts = np.concatenate(points).astype(np.float64)
    if not len(pts):
        return None
    up = np.array([0.0, 1.0, 0.0]) if is_levelled(paths.recon) else estimate_up(
        recon["cam2world"])
    forward = horizontal_view_dir(recon["cam2world"], up)
    right = np.cross(forward, up)
    heights = pts @ up
    floor = float(np.median(heights[heights <= np.percentile(heights, FLOOR_PERCENTILE)]))
    plane = np.stack([pts @ right, pts @ forward], 1)
    lo, hi = np.percentile(plane, [2, 98], axis=0)
    return SceneFrame(up, right, forward, floor, lo, hi)


def box_gap(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Euclidean gap between two axis-aligned boxes [[min], [max]] (0 if they touch)."""
    (alo, ahi), (blo, bhi) = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(alo - bhi, blo - ahi))))


def centroid_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return float(np.linalg.norm(np.subtract(a["centroid"], b["centroid"])))


# ------------------------------------------------------------------ crops / VLM calls


class Crops:
    """Crop images of the changes, loaded on demand from `<out>/<relative path>`."""

    def __init__(self, root: Path):
        self.root = root

    def get(self, change: Mapping[str, Any], key: str) -> tuple[Image.Image, str] | None:
        rel = (change.get("crops") or {}).get(key)
        if not rel or not (self.root / rel).is_file():
            return None
        return Image.open(self.root / rel).convert("RGB"), rel


def owner_side(change: Mapping[str, Any]) -> str:
    """Recording in which the object of a change is visible (moved: before)."""
    if change["type"] == "unverified":
        return change["evidence"]["seen_in"]
    return "after" if change["type"] == "added" else "before"


def fill(prompt: str, label: str) -> str:
    """`prompt` with "{label}" replaced (the JSON examples contain other braces)."""
    return prompt.replace("{label}", label)


def ask_images(vlm: BaseVLM, crops: Sequence[tuple[Image.Image, str] | None], prompt: str,
               tag: str) -> dict[str, Any] | None:
    images = [c for c in crops if c is not None]
    return vlm.ask_json([im for im, _ in images], prompt, tag, [rel for _, rel in images])


def name_change(vlm: BaseVLM, crops: Crops, change: dict[str, Any]) -> None:
    """Ask for label / attributes / description and store them in `change`."""
    if change["type"] == "moved":
        shown = [crops.get(change, "before"), crops.get(change, "after")]
    else:
        shown = [crops.get(change, owner_side(change))]
    shown = [c for c in shown if c is not None]
    if not shown:
        return
    prompt = NAME_TWO if len(shown) == 2 else NAME_ONE
    data = ask_images(vlm, shown, prompt, f"name:{change['id']}") or {}
    change["label"] = normalise_label(data.get("label"))
    change["attributes"] = str(data.get("attributes") or "").strip()
    change["object_description"] = str(data.get("description") or "").strip()


def verdict_of(data: Mapping[str, Any] | None) -> tuple[str, str]:
    """(verdict, reason) from a judge reply; anything unparsable is "unclear"."""
    data = data or {}
    verdict = str(data.get("verdict", "")).strip().lower()
    reason = str(data.get("reason") or "").strip()
    if verdict not in VERDICTS:
        verdict, reason = "unclear", reason or f"unparsable answer: {data!r}"[:200]
    return verdict, reason


def object_verdict(data: Mapping[str, Any] | None) -> tuple[str, str]:
    """Verdict of the object test: rejected whenever the VLM flags the region as inside a
    mirror or as part of the room, whatever its verdict field says."""
    verdict, reason = verdict_of(data)
    data = data or {}
    for key, what in (("inside_mirror", "inside a mirror"), ("part_of_room", "part of the room")):
        if str(data.get(key)).lower() == "true" and verdict != "rejected":
            return "rejected", f"{what}: {reason}"
    return verdict, reason


def judge_change(vlm: BaseVLM, crops: Crops, change: dict[str, Any]) -> tuple[str, str]:
    """Type-specific paired-crop question; returns (verdict, reason). For moved changes
    also sets `motion_kind`. A missing view of the place falls back to the object test."""
    kind, label, cid = change["type"], change["label"], change["id"]
    owner = crops.get(change, owner_side(change))
    if kind == "moved":
        before, after = crops.get(change, "before"), crops.get(change, "after")
        if change["evidence"].get("vlm_verdict") is None:  # not already judged by re-pairing
            data = ask_images(vlm, [before, after], fill(JUDGE_MOVED, label),
                              f"judge_same:{cid}")
            verdict, reason = verdict_of(data)
        else:
            verdict, reason = change["evidence"]["vlm_verdict"], change["evidence"]["vlm_reason"]
        if verdict != "rejected":
            data = ask_images(vlm, [before, after], fill(JUDGE_MOTION, label),
                              f"judge_motion:{cid}") or {}
            motion = str(data.get("motion", "")).strip().lower()
            change["motion_kind"] = motion if motion in MOTION_KINDS else "unknown"
            change["evidence"]["vlm_motion_reason"] = str(data.get("reason") or "").strip()
        return verdict, reason
    place = None
    if kind == "removed":
        place = crops.get(change, "after")
        pair, prompt = [owner, place], fill(JUDGE_REMOVED, label)
    elif kind == "added":
        place = crops.get(change, "before")
        pair, prompt = [place, owner], fill(JUDGE_ADDED, label)
    if kind == "unverified" or place is None:
        verdict, reason = object_verdict(ask_images(vlm, [owner], JUDGE_UNVERIFIED,
                                                    f"judge_object:{cid}"))
        if kind == "unverified" and verdict != "rejected" and is_background(label):
            verdict, reason = "rejected", f"named '{label}' (part of the room): {reason}"
        if kind != "unverified":
            reason = f"(no view of the place in the other video) {reason}"
        return verdict, reason
    return verdict_of(ask_images(vlm, pair, prompt, f"judge_{kind}:{cid}"))


def linked_partner(change: Mapping[str, Any], by_id: Mapping[int, dict[str, Any]]
                   ) -> dict[str, Any] | None:
    """The other change of a candidate in-place replacement (a removed change
    `replaced_by` an added one that `replaces` it), if both are in `by_id` with these
    types; else None."""
    if change["type"] == "removed":
        other = by_id.get(change.get("replaced_by"))
        if other is not None and other["type"] == "added" and \
                other.get("replaces") == change["id"]:
            return other
    elif change["type"] == "added":
        other = by_id.get(change.get("replaces"))
        if other is not None and other["type"] == "removed" and \
                other.get("replaced_by") == change["id"]:
            return other
    return None


def from_appearance(change: Mapping[str, Any]) -> bool:
    """The change's object was found by the appearance channel (`source` of its side)."""
    side = change.get(owner_side(change)) or {}
    return side.get("source") == "appearance"


def in_place_answer(vlm: BaseVLM, shown: Sequence[tuple[Image.Image, str] | None],
                    labels: str, tag: str) -> tuple[str, str] | None:
    """(answer, reason) of `JUDGE_REPLACED` on a before / after crop pair; anything
    unparsable is "unclear"; None if a crop is missing."""
    if None in shown:
        return None
    data = ask_images(vlm, shown, JUDGE_REPLACED.replace("{labels}", labels), tag) or {}
    answer = str(data.get("change", "")).strip().lower()
    reason = str(data.get("reason") or "").strip()
    if answer not in REPLACE_VERDICTS:
        answer, reason = "unclear", reason or f"unparsable answer: {data!r}"[:200]
    return answer, reason


def judge_replaced(vlm: BaseVLM, crops: Crops, removed: Mapping[str, Any],
                   added: Mapping[str, Any]) -> tuple[str, str] | None:
    """In-place question on the owner crops of a candidate replacement pair (removed
    object in image 1, added object in image 2); None if a crop is missing."""
    labels = (f"the region is probably a {removed['label']} in image 1 and a "
              f"{added['label']} in image 2; a coloured outline traces it")
    return in_place_answer(vlm, [crops.get(removed, "before"), crops.get(added, "after")],
                           labels, f"judge_replaced:{removed['id']}+{added['id']}")


def judge_in_place(vlm: BaseVLM, crops: Crops, change: Mapping[str, Any]
                   ) -> tuple[str, str] | None:
    """In-place question for a single removed / added appearance change: its owner crop
    and the other recording's view of the place, in time order; None if one is missing."""
    owner = "before" if change["type"] == "removed" else "after"
    image = 1 if owner == "before" else 2
    labels = (f"the region in image {image} is probably a {change['label']}, traced by a "
              f"coloured outline; image {3 - image} shows the same place")
    shown = [crops.get(change, "before"), crops.get(change, "after")]
    return in_place_answer(vlm, shown, labels, f"judge_in_place:{change['id']}")


def surface_rejection(change: Mapping[str, Any]) -> str | None:
    """Reason to reject an appearance-channel change whose label names part of the room
    or a bare surface (`is_background`, `SURFACE_NOUNS`), else None."""
    label = change["label"]
    if from_appearance(change) and (is_background(label) or head_noun(label) in SURFACE_NOUNS):
        return f"named '{label}' (part of the room or a bare surface)"
    return None


def apply_verdict(change: dict[str, Any], verdict: str, reason: str,
                  unclear_factor: float) -> bool:
    """Store the verdict in `change`; returns False if the change is rejected."""
    change["evidence"]["vlm_verdict"] = verdict
    change["evidence"]["vlm_reason"] = reason
    if verdict == "unclear":
        change["confidence"] = round(change["confidence"] * unclear_factor, 3)
    return verdict != "rejected"


# ------------------------------------------------------------------ re-pairing


def merge_removed_added(r: Mapping[str, Any], a: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """One `moved` change from a removed change `r` and an added / after-only unverified
    change `a` of the same object (translation from the centroids, rotation unknown)."""
    b_side, a_side = r["before"], a["after"]
    t = np.subtract(a_side["centroid"], b_side["centroid"])
    ev_r, ev_a = r["evidence"], a["evidence"]
    views_b, views_a = ev_r.get("observing_views", 0), ev_a.get("observing_views", 0)
    crops_r, crops_a = r.get("crops") or {}, a.get("crops") or {}
    return {
        "id": r["id"], "type": "moved", "label": r["label"],
        "attributes": r.get("attributes", ""),
        "object_description": r.get("object_description", ""),
        "confidence": round((r["confidence"] + a["confidence"]) / 2, 3),
        "before": b_side, "after": a_side,
        "transform": {"translation_m": round(float(np.linalg.norm(t)), 4),
                      "rotation_deg": None, "rotation_reliable": False, "R": np.eye(3).tolist(),
                      "t": t.round(5).tolist(), "icp_fitness": None, "icp_rmse": None},
        "motion_kind": None,
        "evidence": {
            "free_rays": ev_r.get("free_rays", 0) + ev_a.get("free_rays", 0),
            "observing_views": round((views_b + views_a) / 2),
            "observing_views_before_place": views_b,
            "observing_views_after_place": views_a,
            "appearance_similarity": None,
            "mask_score": round((ev_r.get("mask_score", 0) + ev_a.get("mask_score", 0)) / 2, 4),
            "merged_from": [r["id"], a["id"]],
            "merged_types": [r["type"], a["type"]],
            "vlm_verdict": "confirmed", "vlm_reason": reason,
        },
        "description": "",
        "crops": {"before": crops_r.get("before"), "after": crops_a.get("after"),
                  "before_place": crops_r.get("after"), "after_place": crops_a.get("before")},
    }


def side_instance_ids(side: Mapping[str, Any]) -> list[int]:
    """Instance ids of a change side (several after a merge of parts)."""
    return list(side.get("instance_ids") or [side["instance_id"]])


def union_side(side: Mapping[str, Any], part: Mapping[str, Any],
               points: np.ndarray | None = None) -> dict[str, Any]:
    """`side` grown by `part` (two parts of one object in one recording): box, centroid
    and oriented box of the union of their instance `points` if given, else the union of
    the boxes with the point-weighted centroid (the oriented box encloses both `obb`s);
    `n_points` and `blob_ids` merged, `instance_ids` of both."""
    if points is not None and len(points):
        pts = np.asarray(points, np.float64)
        lo, hi, centroid, obb, n = pts.min(0), pts.max(0), pts.mean(0), floor_obb(pts), len(pts)
    else:
        lo = np.minimum(side["aabb"][0], part["aabb"][0])
        hi = np.maximum(side["aabb"][1], part["aabb"][1])
        n1, n2 = side.get("n_points", 1), part.get("n_points", 1)
        centroid = (np.multiply(side["centroid"], n1) + np.multiply(part["centroid"], n2)) / (
            n1 + n2)
        obb = floor_obb(np.vstack([obb_corners(side["obb"]), obb_corners(part["obb"])])) \
            if side.get("obb") and part.get("obb") else None
        n = n1 + n2
    return {**side, "centroid": np.round(centroid, 4).tolist(),
            "aabb": [np.round(lo, 4).tolist(), np.round(hi, 4).tolist()],
            "extents": np.round(np.subtract(hi, lo), 4).tolist(), "obb": obb, "n_points": int(n),
            "blob_ids": sorted(set(side.get("blob_ids", [])) | set(part.get("blob_ids", []))),
            "instance_ids": side_instance_ids(side) + side_instance_ids(part)}


def absorb_part(m: dict[str, Any], a: Mapping[str, Any], reason: str) -> None:
    """Union the after side of the added change `a` into the moved change `m` (two parts
    of one deformable object; the oriented box encloses both boxes); the transform becomes
    the centroid translation."""
    m["after"] = union_side(m["after"], a["after"])
    centroid = np.asarray(m["after"]["centroid"])
    t = centroid - np.asarray(m["before"]["centroid"])
    m["transform"] = {**m["transform"], "translation_m": round(float(np.linalg.norm(t)), 4),
                      "rotation_deg": None, "rotation_reliable": False, "R": np.eye(3).tolist(),
                      "t": t.round(5).tolist()}
    m["confidence"] = round((m["confidence"] + a["confidence"]) / 2, 3)
    ev = m["evidence"]
    ev["free_rays"] = ev.get("free_rays", 0) + a["evidence"].get("free_rays", 0)
    ev["merged_from"] = [*ev.get("merged_from", [m["id"]]), a["id"]]
    ev["merged_part_reason"] = reason
    m["crops"] = {**m["crops"], "after_part": (a.get("crops") or {}).get("after")}


def change_gap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Gap between two change sides: between their oriented boxes, else their AABBs."""
    if a.get("obb") and b.get("obb"):
        return obb_gap(a["obb"], b["obb"])
    return box_gap(a["aabb"], b["aabb"])


def merge_fragment_pair(
    x: Mapping[str, Any], y: Mapping[str, Any], name: str,
    points: Mapping[str, Mapping[int, np.ndarray]], reason: str, label: str | None, gap: float,
) -> dict[str, Any]:
    """One change from two changes `x`, `y` of one type showing parts of one object (side
    `name`: before for removed, after for added). The fragment with more points keeps its
    id, label (unless the VLM named the whole object: `label`), attributes, crops (+ the
    other's owner crop as `<name>_part`) and evidence (free rays and occupied points summed,
    observing views max); geometry from the union of both fragments' instance `points`
    (box union without them); confidence = max; description regenerated later."""
    big, small = (x, y) if x[name].get("n_points", 0) >= y[name].get("n_points", 0) else (y, x)
    own = [points.get(name, {}).get(i) for i in side_instance_ids(big[name])
           + side_instance_ids(small[name])]
    union = np.concatenate(own) if all(p is not None and len(p) for p in own) else None
    merged = json.loads(json.dumps(big))  # deep copy
    merged[name] = union_side(big[name], small[name], union)
    merged["label"] = label or big["label"]
    merged["confidence"] = max(big["confidence"], small["confidence"])
    ev, other = merged["evidence"], small["evidence"]
    for key in ("free_rays", "occ_points"):
        if key in ev or key in other:
            ev[key] = ev.get(key, 0) + other.get(key, 0)
    ev["observing_views"] = max(ev.get("observing_views", 0), other.get("observing_views", 0))
    ev["merged_from"] = [big["id"], small["id"]]
    ev["merged_types"] = [big["type"], small["type"]]
    ev["merged_part_reason"] = reason
    ev["fragment_gap_m"] = round(gap, 3)
    merged["crops"] = {**(big.get("crops") or {}),
                       f"{name}_part": (small.get("crops") or {}).get(name)}
    merged["description"] = ""
    return merged


def merge_fragments(vlm: BaseVLM, crops: Crops, changes: list[dict[str, Any]],
                    params: Mapping[str, Any], points: Mapping[str, Mapping[int, np.ndarray]],
                    only: set[int] | None = None,
                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge the fragments of one object (module docstring, step 2): pairs of removed /
    removed or added / added changes with matching labels whose oriented boxes are at
    most `fragment_gap_m` apart, closest first, each change in at most one merge, if the
    VLM says both owner crops show parts of one physical object. `only`: consider just
    the pairs with one of these ids. Returns the new change list and one record per VLM
    pair check."""
    by_id = {c["id"]: c for c in changes}
    pairs = []
    for kind, name in (("removed", "before"), ("added", "after")):
        group = [c for c in changes if c["type"] == kind and c.get(name)]
        for k, x in enumerate(group):
            for y in group[k + 1:]:
                if only is not None and x["id"] not in only and y["id"] not in only:
                    continue
                if labels_match(x["label"], y["label"]):
                    gap = change_gap(x[name], y[name])
                    if gap <= params["fragment_gap_m"]:
                        pairs.append((gap, x["id"], y["id"], name))
    records: list[dict[str, Any]] = []
    used: set[int] = set()
    for gap, xid, yid, name in sorted(pairs):
        x, y = by_id[xid], by_id[yid]
        shown = [crops.get(x, name), crops.get(y, name)]
        if xid in used or yid in used or None in shown:
            continue
        data = ask_images(vlm, shown, fill(JUDGE_FRAGMENTS, x["label"]),
                          f"fragments:{xid}+{yid}")
        verdict, reason = verdict_of(data)
        label = normalise_label((data or {}).get("label"))
        if not (labels_match(label, x["label"]) or labels_match(label, y["label"])):
            label = None  # keep the larger fragment's name
        records.append({"kind": f"{x['type']}+{y['type']}", "ids": [xid, yid],
                        "labels": [x["label"], y["label"]], "obb_gap_m": round(gap, 3),
                        "verdict": verdict, "reason": reason, "label": label})
        if verdict == "confirmed":
            merged = merge_fragment_pair(x, y, name, points, reason, label, gap)
            by_id[merged["id"]] = merged
            del by_id[merged["evidence"]["merged_from"][1]]
            used |= {xid, yid}
    return list(by_id.values()), records


def repair(vlm: BaseVLM, crops: Crops, changes: list[dict[str, Any]],
           params: Mapping[str, Any], only: set[int] | None = None,
           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-pair split changes (module docstring, step 2). `only`: consider just the pairs
    with one of these ids, never two halves of the same rejected moved change. Returns the
    new change list and one record per VLM pair check."""
    records: list[dict[str, Any]] = []
    by_id = {c["id"]: c for c in changes}

    def wanted(x: Mapping[str, Any], y: Mapping[str, Any]) -> bool:
        if only is None:
            return True
        same_split = x["evidence"].get("split_from") is not None and \
            x["evidence"].get("split_from") == y["evidence"].get("split_from")
        return (x["id"] in only or y["id"] in only) and not same_split

    removed = [c for c in changes if c["type"] == "removed"]
    targets = [c for c in changes if c["type"] == "added"
               or (c["type"] == "unverified" and c["evidence"].get("seen_in") == "after")]
    pairs = sorted((centroid_distance(r["before"], a["after"]), r["id"], a["id"])
                   for r in removed for a in targets
                   if labels_match(r["label"], a["label"]) and wanted(r, a)
                   and r.get("replaced_by") != a["id"])  # not the two sides of one place
    used: set[int] = set()
    for dist, rid, aid in pairs:
        if dist >= params["repair_max_distance"] or rid in used or aid in used:
            continue
        r, a = by_id[rid], by_id[aid]
        data = ask_images(vlm, [crops.get(r, "before"), crops.get(a, "after")],
                          fill(JUDGE_SAME, r["label"]), f"repair:{rid}+{aid}")
        verdict, reason = verdict_of(data)
        records.append({"kind": "removed+" + a["type"], "ids": [rid, aid],
                        "labels": [r["label"], a["label"]], "distance_m": round(dist, 3),
                        "verdict": verdict, "reason": reason})
        if verdict == "confirmed":
            by_id[rid] = merge_removed_added(r, a, reason)
            del by_id[aid]
            used |= {rid, aid}

    moved = [c for c in by_id.values() if c["type"] == "moved"]
    added = [c for c in by_id.values() if c["type"] == "added"]
    pairs = sorted((change_gap(m["after"], a["after"]), m["id"], a["id"])
                   for m in moved for a in added
                   if labels_match(m["label"], a["label"]) and wanted(m, a))
    for gap, mid, aid in pairs:
        if gap > params["merge_max_distance"] or aid not in by_id or mid not in by_id:
            continue
        m, a = by_id[mid], by_id[aid]
        data = ask_images(vlm, [crops.get(m, "after"), crops.get(a, "after")],
                          fill(JUDGE_PARTS, m["label"]), f"merge_part:{mid}+{aid}")
        verdict, reason = verdict_of(data)
        records.append({"kind": "moved+added", "ids": [mid, aid],
                        "labels": [m["label"], a["label"]], "box_gap_m": round(gap, 3),
                        "centroid_distance_m": round(centroid_distance(m["after"], a["after"]), 3),
                        "verdict": verdict, "reason": reason})
        if verdict == "confirmed":
            absorb_part(m, a, reason)
            del by_id[aid]
    return list(by_id.values()), records


def split_moved(
    m: Mapping[str, Any], reason: str, new_id: int,
    records: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """A moved change `m` whose match the judge rejected, as a `removed` change (its before
    side) and an `added` change (its after side), with `m`'s confidence, not yet judged
    (`evidence.split_reason` keeps the rejected match). Ids: the merged candidates' ids
    for a re-paired change, else `m`'s id and `new_id`. Evidence from the instance
    `records` (per recording and id) when available, else from `m`'s per-place evidence."""
    ev, crops = m["evidence"], m.get("crops") or {}
    ids = ev.get("merged_from") or []
    ids = ids[:2] if len(ids) >= 2 else [m["id"], new_id]
    halves = []
    for cid, kind, name, own, other in ((ids[0], "removed", "before", "before", "before_place"),
                                        (ids[1], "added", "after", "after", "after_place")):
        side = m[name]
        record = records.get(name, {}).get(side.get("instance_id"))
        evidence = instance_evidence(record) if record else {
            "observing_views": ev.get(f"observing_views_{name}_place", 0),
            "mask_score": ev.get("mask_score", 0.0)}
        crop_other = "after" if name == "before" else "before"
        halves.append({
            "id": cid, "type": kind, "label": m["label"],
            "attributes": m.get("attributes", ""),
            "object_description": m.get("object_description", ""),
            "confidence": m["confidence"],
            "before": side if name == "before" else None,
            "after": side if name == "after" else None,
            "transform": None, "motion_kind": None,
            "evidence": {**evidence, "split_from": m["id"],
                         "split_reason": f"appearance judge did not confirm the match: {reason}",
                         "vlm_verdict": None, "vlm_reason": None},
            "description": "",
            "crops": {name: crops.get(own), crop_other: crops.get(other)},
        })
    return halves


# ------------------------------------------------------------------ descriptions


def size_text(extents: Sequence[float]) -> str:
    """"about 0.9 x 0.5 x 0.4 m" (sorted, one decimal), in cm below 0.3 m."""
    dims = sorted((max(float(e), 0.01) for e in extents), reverse=True)
    if dims[0] < 0.3:
        return "about " + " x ".join(f"{100 * d:.0f}" for d in dims) + " cm"
    return "about " + " x ".join(f"{max(d, 0.1):.1f}" for d in dims) + " m"


def side_extents(side: Mapping[str, Any]) -> Sequence[float]:
    """Size of a change side: the oriented box extents if present, else the AABB's."""
    return (side.get("obb") or {}).get("extents") or side["extents"]


def height_text(h: float, on_floor: str = "on the floor") -> str:
    """`on_floor` below ON_FLOOR_M, else "0.4 m above the floor"."""
    return on_floor if h < ON_FLOOR_M else f"{h:.1f} m above the floor"


def where(side: Mapping[str, Any], frame: SceneFrame | None, pts: np.ndarray | None,
          on_floor: str = "on the floor") -> tuple[str, str]:
    """("0.5 m above the floor", "in the far left part of the room") for one change side
    (`pts`: its instance points; the box corners if None). Without a scene frame:
    ("", "at (x, y, z)")."""
    if frame is None:
        return "", "at (" + ", ".join(f"{v:.1f}" for v in side["centroid"]) + ")"
    if pts is None or not len(pts):
        lo, hi = np.asarray(side["aabb"], np.float64)
        pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    return height_text(frame.height(pts), on_floor), frame.place(side["centroid"])


def article(label: str) -> str:
    """"A" / "An", or "The" for a plural-looking label ("pajamas")."""
    if head_noun(label) != label.split()[-1]:
        return "The"
    return "An" if label[:1] in "aeiou" else "A"


def template_description(change: Mapping[str, Any], frame: SceneFrame | None,
                         points: Mapping[str, Mapping[int, np.ndarray]],
                         partner: Mapping[str, Any] | None = None) -> str:
    """One templated sentence: label, type, geometry and evidence. `partner`: the kept
    other change of an in-place replacement (the removed object was replaced by it)."""
    label, kind, ev = change["label"], change["type"], change["evidence"]
    attrs = change.get("attributes") or ""

    def side_where(name: str, on_floor: str = "on the floor") -> tuple[str, str]:
        side = change[name]
        ids = side.get("instance_ids") or [side.get("instance_id")]
        own = [points.get(name, {}).get(i) for i in ids]
        own = [p for p in own if p is not None and len(p)]
        return where(side, frame, np.concatenate(own) if own else None, on_floor)

    def joined(name: str) -> str:
        return " ".join(x for x in side_where(name) if x)

    def views(n: Any) -> str:
        n = int(n or 0)
        return f"{n} viewpoint{'s' if n != 1 else ''}"

    if kind == "moved":
        tf = change["transform"] or {}
        size = size_text(side_extents(change["before"]))
        what = f"The {label}" + (f" ({attrs}; {size})" if attrs else f" ({size})")
        (h_b, p_b), (h_a, p_a) = side_where("before"), side_where("after")
        text = f"{what} moved {tf.get('translation_m', 0):.1f} m"
        if p_b == p_a:  # same coarse region: say it once
            text += " within " + p_b.split(" ", 1)[1]
        else:
            text += f", from {p_b.split(' ', 1)[1]} to {p_a.split(' ', 1)[1]}"
        if h_b and h_a:  # the lowest observed part, not necessarily the whole object
            text += (f" ({h_b} both times)" if h_b == h_a else
                     f" (lowest part {h_b} before, {h_a} after)")
        if change.get("motion_kind") == "deformed":
            text += ", mostly by being rearranged or deformed rather than carried as a whole"
        elif tf.get("rotation_deg") is not None and tf["rotation_deg"] >= 10 and \
                tf.get("rotation_reliable", True):  # an unreliable angle is not quoted
            text += f" and turned by about {tf['rotation_deg']:.0f} degrees"
        nb, na = ev.get("observing_views_before_place", 0), ev.get("observing_views_after_place", 0)
        text += f"; the after recording saw the old place empty from {views(nb)}"
        text += (f" and the before recording saw the new place empty from {views(na)}" if na
                 else " (the before recording never looked at the new place)")
        if len(ev.get("merged_from", [])) > 1:
            text += f"; this entry merges candidates {', '.join(map(str, ev['merged_from']))}"
        return text + "."
    side = owner_side(change)
    size = size_text(side_extents(change[side]))
    what = f"{label}" + (f" ({attrs}; {size})" if attrs else f" ({size})")
    if kind == "removed" and partner is not None:
        return (f"The {what}, which was {joined('before')}, was replaced by "
                f"{article(partner['label']).lower()} {partner['label']}: the after recording "
                f"looked at that spot from {views(ev.get('observing_views'))} and saw the "
                f"{partner['label']} there instead.")
    if kind == "added" and partner is not None:
        return (f"{article(label)} {what} took the place of the {partner['label']} "
                f"{joined('after')}: the before recording looked at that spot from "
                f"{views(ev.get('observing_views'))} and saw the {partner['label']} there.")
    if kind == "removed":
        return (f"The {what}, which was {joined('before')}, was removed: the after recording "
                f"looked at that spot from {views(ev.get('observing_views'))} and saw it empty.")
    if kind == "added":
        return (f"{article(label)} {what} appeared {joined('after')}: the before recording "
                f"looked at that spot from {views(ev.get('observing_views'))} and saw it empty.")
    other = "after" if side == "before" else "before"
    return (f"{article(label)} {what}, {joined(side)}, was seen only in the {side} "
            f"recording; the {other} recording never looked at that spot, so this possible "
            "change cannot be verified.")


def numbers_preserved(template: str, text: str, label: str) -> bool:
    """Every number of `template` and the label's head noun also appear in `text`."""
    found = set(NUMBER.findall(text))
    return (bool(text.strip()) and set(NUMBER.findall(template)) <= found
            and head_noun(label) in text.lower())


def polish(vlm: BaseVLM, sentences: Sequence[str], labels: Sequence[str]) -> list[str | None]:
    """VLM-polished version of each sentence, or None where numbers / label got lost."""
    if not sentences:
        return []
    items = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(sentences))
    budget = min(2048, max(vlm.max_new_tokens, 90 * len(sentences)))
    data = vlm.ask_json([], POLISH.format(items=items, n=len(sentences)), "polish",
                        max_new_tokens=budget) or {}
    out = data.get("sentences")
    if not isinstance(out, list) or len(out) != len(sentences):
        return [None] * len(sentences)
    return [str(new).strip() if numbers_preserved(old, str(new), label) else None
            for old, new, label in zip(sentences, out, labels)]


def summary_line(changes: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(c["type"] for c in changes)
    verified = sum(counts[t] for t in ("moved", "added", "removed"))
    return (f"{verified} change{'s' if verified != 1 else ''}: {counts['moved']} moved, "
            f"{counts['added']} added, {counts['removed']} removed; "
            f"{counts['unverified']} unverified")


# ------------------------------------------------------------------ stage


def _peak_memory() -> dict[str, Any]:
    torch = sys.modules.get("torch")
    vram = None
    if torch is not None and torch.cuda.is_initialized():
        vram = {"allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    return {"peak_vram_gb": vram,
            "peak_ram_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}


def load_m4_changes(paths: OutputPaths) -> dict[str, Any]:
    """The M4 changes document: `changes.json` if it has not been through M5 yet (it is
    then backed up to `semantics/changes_m4.json`), else the backup."""
    backup = paths.root / "semantics" / "changes_m4.json"
    doc = json.loads(paths.changes_json.read_text())
    if "rejected" not in doc:
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.changes_json, backup)
        return doc
    if not backup.is_file():
        raise FileNotFoundError(f"{paths.changes_json} was already processed by M5 and "
                                f"{backup} is missing; re-run `roomdiff changes` first")
    return json.loads(backup.read_text())


def instance_points(paths: OutputPaths) -> dict[str, dict[int, np.ndarray]]:
    """Instance points per recording and id (empty if the M4 files are missing)."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for name in RECORDINGS:
        file = paths.instances / f"{name}_points.npz"
        if file.is_file():
            with np.load(file) as f:
                pts, owner = f["points"], f["instance"]
            out[name] = {int(i): pts[owner == i] for i in np.unique(owner)}
    return out


def instance_records(paths: OutputPaths) -> dict[str, dict[int, dict[str, Any]]]:
    """M4 instance records per recording and id (empty if the files are missing)."""
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for name in RECORDINGS:
        file = paths.instances / f"{name}.json"
        if file.is_file():
            out[name] = {r["id"]: r for r in json.loads(file.read_text())["instances"]}
    return out


def sort_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {t: k for k, t in enumerate(CHANGE_TYPES)}
    return sorted(changes, key=lambda c: (order[c["type"]], -c["confidence"], c["id"]))


def run_semantics(out_dir: str | Path, cfg: Mapping[str, Any],
                  vlm: BaseVLM | None = None) -> dict[str, Any]:
    """Name, judge, re-pair and describe the changes of `<out>/changes.json` (module
    docstring). Loads Qwen3-VL unless `vlm` is given. Returns the new changes document."""
    params = dict(cfg["semantics"])
    paths = OutputPaths(out_dir)
    folder = paths.root / "semantics"
    times: dict[str, float] = {}
    start = time.perf_counter()
    doc = load_m4_changes(paths)
    changes = [json.loads(json.dumps(c)) for c in doc["changes"]]  # deep copy
    counts_in = dict(Counter(c["type"] for c in changes))
    for c in changes:
        c.setdefault("attributes", "")
        c.setdefault("object_description", "")
        c["motion_kind"] = None
        c["evidence"]["vlm_verdict"] = None
        c["evidence"]["vlm_reason"] = None
    frame = scene_frame(paths)
    points = instance_points(paths)
    crops = Crops(paths.root)
    times["load"] = time.perf_counter() - start

    t0 = time.perf_counter()
    if vlm is None and changes:
        from roomdiff.vlm import load_vlm

        vlm = load_vlm(params)
    times["load_model"] = time.perf_counter() - t0
    if vlm is None:  # no changes: no VLM call is made, a fake keeps the code path uniform
        from roomdiff.vlm import FakeVLM

        vlm = FakeVLM()
    vlm.open_log(folder / "log.jsonl")

    t0 = time.perf_counter()
    for c in changes:
        name_change(vlm, crops, c)
    times["naming"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    merges: list[dict[str, Any]] = []
    if params["enable_fragment_merge"]:
        changes, merges = merge_fragments(vlm, crops, changes, params, points)
    times["fragments"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if params["enable_repair"]:
        changes, more = repair(vlm, crops, changes, params)
        merges += more
    times["repair"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    kept, rejected, splits, replacements = [], [], [], []
    records = instance_records(paths)
    next_id = max([c["id"] for c in doc["changes"]], default=0) + 1
    by_id = {c["id"]: c for c in changes}
    replace_verdicts: dict[int, tuple[str, str, str]] = {}  # id -> answer, verdict, reason
    for c in changes:
        if not params["enable_judge"]:
            kept.append(c)
            continue
        partner = linked_partner(c, by_id)
        if partner is not None and c["id"] not in replace_verdicts:
            removed, added = (c, partner) if c["type"] == "removed" else (partner, c)
            judged = judge_replaced(vlm, crops, removed, added)
            if judged is not None:  # else each side is judged on its own
                answer, why = judged
                replacements.append({"ids": [removed["id"], added["id"]],
                                     "labels": [removed["label"], added["label"]],
                                     "answer": answer, "reason": why})
                for side, verdict in zip((removed, added), REPLACE_VERDICTS[answer]):
                    replace_verdicts[side["id"]] = (answer, verdict, f"in place: {answer}; {why}")
        if c["id"] not in replace_verdicts and partner is None and \
                c["type"] in ("removed", "added") and from_appearance(c):
            judged = judge_in_place(vlm, crops, c)
            if judged is not None:  # else the usual removed / added question
                answer, why = judged
                verdict = REPLACE_VERDICTS[answer][0 if c["type"] == "removed" else 1]
                replace_verdicts[c["id"]] = (answer, verdict, f"in place: {answer}; {why}")
        if c["id"] in replace_verdicts:
            answer, verdict, reason = replace_verdicts[c["id"]]
            c["evidence"]["vlm_replace_answer"] = answer
            surface = surface_rejection(c)
            if surface is not None and verdict != "rejected":
                verdict, reason = "rejected", f"{surface}: {reason}"
            keep = apply_verdict(c, verdict, reason, params["unclear_factor"])
            (kept if keep else rejected).append(c)
            continue
        verdict, reason = judge_change(vlm, crops, c)
        surface = surface_rejection(c)
        if surface is not None and verdict != "rejected":
            verdict, reason = "rejected", f"{surface}: {reason}"
        if c["type"] == "moved" and verdict == "rejected":  # keep the geometry, unpaired
            halves = split_moved(c, reason, next_id, records)
            if halves[1]["id"] == next_id:
                next_id += 1
            for half in halves:  # named and judged like any removed / added change
                name_change(vlm, crops, half)
                keep = apply_verdict(half, *judge_change(vlm, crops, half), params["split_factor"])
                (kept if keep else rejected).append(half)
            splits.append({"id": c["id"], "label": c["label"], "into": [h["id"] for h in halves],
                           "labels": [h["label"] for h in halves], "reason": reason,
                           "verdicts": [h["evidence"]["vlm_verdict"] for h in halves]})
            continue
        keep = apply_verdict(c, verdict, reason, params["unclear_factor"])
        (kept if keep else rejected).append(c)
    halves = {i for x in splits for i in x["into"]}
    if splits and params["enable_fragment_merge"]:  # a kept half may be part of another change
        kept, more = merge_fragments(vlm, crops, kept, params, points, only=halves)
        merges += more
        halves |= {i for x in more if x["verdict"] == "confirmed" for i in x["ids"]}
    if splits and params["enable_repair"]:  # a kept split half may pair up with another change
        kept, more = repair(vlm, crops, kept, params, only=halves)
        merges += more
        for c in kept:
            if c["type"] == "moved" and c["motion_kind"] is None:  # new: verdict from repair
                verdict, reason = judge_change(vlm, crops, c)  # asks for the motion kind
                apply_verdict(c, verdict, reason, params["unclear_factor"])
    kept, rejected = sort_changes(kept), sort_changes(rejected)
    times["judge"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    kept_by_id = {c["id"]: c for c in kept}
    templates = [template_description(c, frame, points, linked_partner(c, kept_by_id))
                 for c in kept]
    polished = polish(vlm, templates, [c["label"] for c in kept]) if params["enable_polish"] \
        else [None] * len(kept)
    for c, tpl, new in zip(kept, templates, polished):
        c["description"] = new or tpl
        c["description_template"] = tpl
    times["describe"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    line = summary_line(kept)
    paragraph = ""
    if kept:
        items = "\n".join(f"- [{c['type']}] {c['description']}" for c in kept)
        data = vlm.ask_json([], SUMMARY.format(items=items), "summary") or {}
        paragraph = str(data.get("summary") or "").strip()
    else:
        paragraph = "No change between the two videos was confirmed."
    times["summary"] = time.perf_counter() - t0

    result = {"scene": doc["scene"], "changes": kept, "rejected": rejected,
              "summary": f"{line}. {paragraph}".strip()}
    paths.changes_json.write_text(json.dumps(result, indent=2))
    times["total"] = time.perf_counter() - start
    meta = {
        "model": vlm.model_id if vlm.calls else params["model"],
        "params": params,
        "runtime_s": {k: round(v, 2) for k, v in times.items()},
        "vlm_calls": vlm.calls,
        "vlm_seconds": round(vlm.seconds, 2),
        "counts_in": counts_in,
        "counts_out": dict(Counter(c["type"] for c in kept)),
        "rejected": [{"id": c["id"], "type": c["type"], "label": c["label"],
                      "reason": c["evidence"]["vlm_reason"]} for c in rejected],
        "merges": merges,
        "splits": splits,
        "replacements": replacements,
        "verdicts": {str(c["id"]): c["evidence"]["vlm_verdict"] for c in kept + rejected},
        "polished": sum(p is not None for p in polished),
        "template_kept": sum(p is None for p in polished),
        "scene_frame": None if frame is None else {
            "up": frame.up.round(4).tolist(), "floor_height": round(frame.floor, 3)},
        **_peak_memory(),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[semantics] {result['summary'][:200]} | {len(rejected)} rejected, "
          f"{len([m for m in merges if m['verdict'] == 'confirmed'])} merged, "
          f"{len(splits)} moved split, "
          f"{vlm.calls} VLM calls, {times['total']:.1f} s -> {paths.changes_json}")
    return result
