"""Score a RoomDiff run against the hand-written ground truth of a scene (gt.yaml).

Usage: python scripts/score_gt.py <out_dir> <gt.yaml> [--json] [--threshold 0.6] [--scale S]

Schema: docs/gt_template.yaml, documented in docs/capture_protocol.md. Every GT change is
matched to an unused prediction of `<out_dir>/changes.json` (not `rejected`) of the same
type whose label matches; among several, the most confident wins. A GT `moved` object
left unmatched then accepts a removed + added pair whose labels both match ("type wrong";
displacement = distance of the two centroids); a single matching half is "partial" and
counts as missed. A GT entry may list `aliases` (other names of the same object); its best-matching
name counts. Objects under `unchanged_but_only_in_one_video` must not come back as
removed / added / moved (reported as unverified, or not at all, is correct). Unmatched
predictions are extra (false positives); unmatched unverified ones are listed apart, as
they claim no change.

Labels match when their token overlap reaches --threshold: lowercase, cut at a
preposition ("lamp on the desk" -> "lamp"), articles dropped, colours / materials / sizes
dropped unless they are the last word, words singularised, synonyms mapped to one word
(chair / seat, bag / backpack, box / carton, sofa / couch, jumper / sweater, trash can /
bin, pyjamas / pajamas, ...). The last word (head noun) weighs 2, the others 1;
score = shared weight / total weight of the shorter label, so "office chair" ~ "chair"
(1.0), "office chair" ~ "desk chair" (0.67), "coffee table" !~ "coffee mug" (0.33).

`--scale` multiplies the predicted distances (e.g. the metric scale of
scripts/lidar_check.py). Prints a markdown report; `--json` also writes
`<out_dir>/gt_score.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

TYPES = ("moved", "added", "removed")
TOP_KEYS = {"scene", "notes", "changes", "unchanged_but_only_in_one_video"}
CHANGE_KEYS = {"object", "aliases", "type", "displacement_m", "rotation_deg", "size_m", "where",
               "notes"}
UNCHANGED_KEYS = {"object", "aliases", "seen_in", "where", "notes"}
STOP = {"a", "an", "the", "some", "of", "and", "pair", "piece"}
CUT = {"on", "in", "with", "against", "at", "near", "under", "behind", "from", "beside",
       "by", "next", "inside", "onto", "above", "below", "underneath", "between", "into"}
MODIFIERS = {  # dropped unless last: colours, materials, sizes
    "red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "brown", "black",
    "white", "grey", "gray", "beige", "cream", "silver", "gold", "golden", "navy", "teal",
    "turquoise", "maroon", "khaki", "tan", "dark", "pale", "bright", "colourful", "colorful",
    "striped", "patterned", "wooden", "wood", "metal", "metallic", "steel", "plastic",
    "fabric", "leather", "cotton", "wool", "woollen", "woolen", "glass", "cardboard",
    "paper", "ceramic", "wicker", "rattan", "velvet", "denim", "mesh", "stone", "marble",
    "bamboo", "rubber", "small", "large", "big", "little", "tiny", "huge", "tall", "medium"}
GENERIC = {"object", "thing", "item", "stuff"}
PHRASES = {  # applied to the singularised text
    "trash can": "bin", "garbage can": "bin", "waste basket": "bin", "wastepaper basket": "bin",
    "rubbish bin": "bin", "trash bin": "bin", "waste bin": "bin", "laptop computer": "laptop",
    "remote control": "remote", "teddy bear": "plush", "stuffed animal": "plush",
    "stuffed toy": "plush", "soft toy": "plush", "plush toy": "plush", "potted plant": "plant",
    "house plant": "plant", "pot plant": "plant"}
SYNONYMS = {  # one word for each group of names of the same kind of object
    "chair": {"seat", "armchair", "stool"}, "sofa": {"couch", "settee"},
    "bag": {"backpack", "rucksack", "handbag", "purse", "tote", "satchel"},
    "box": {"carton", "package", "parcel"}, "sweater": {"jumper", "pullover", "sweatshirt"},
    "bin": {"wastebasket", "dustbin", "trashcan"}, "pajama": {"pyjama"}, "pillow": {"cushion"},
    "cup": {"mug"}, "rug": {"carpet", "mat"}, "blanket": {"throw", "duvet", "quilt", "comforter"},
    "tv": {"television", "telly"}, "jacket": {"coat"}, "shoe": {"sneaker", "trainer"},
    "plant": {"houseplant"}, "tin": {"can", "canister", "caddy"}, "bottle": {"jug", "flask"},
    "bucket": {"pail"}, "dryer": {"hairdryer", "blowdryer"},
}
CANONICAL = {word: key for key, words in SYNONYMS.items() for word in words}


class GTError(ValueError):
    """Malformed gt.yaml."""


# ------------------------------------------------------------------------ labels


def stem(word: str) -> str:
    """Naive singular, made comparable: "boxes" / "box", "shelves" / "shelf", "hoodies" /
    "hoodie" map to one key."""
    if len(word) > 3 and not word.endswith(("ss", "us", "is")):
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith(("ches", "shes", "xes", "zes", "sses")):
            word = word[:-2]
        elif word.endswith("s"):
            word = word[:-1]
    if len(word) > 3 and word.endswith("ie"):
        word = word[:-2] + "y"
    if len(word) > 3 and word.endswith(("ve", "fe")):
        word = word[:-2] + "f"
    return word


def label_tokens(text: Any) -> list[str]:
    """Normalised words of a label; the last one is the head noun."""
    words = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower().replace("-", " ")).split()
    cut = next((k for k, w in enumerate(words) if w in CUT and k > 0), len(words))
    words = [w for w in words[:cut] if w not in STOP]
    words = [w for k, w in enumerate(words) if w not in MODIFIERS or k == len(words) - 1]
    text = " ".join(stem(w) for w in words)
    for phrase, word in PHRASES.items():
        text = re.sub(rf"\b{phrase}\b", word, text)
    return [CANONICAL.get(w, w) for w in text.split() if w not in GENERIC]


def label_score(a: list[str], b: list[str]) -> float:
    """Weighted token overlap in [0, 1] (head noun weight 2, other words 1)."""
    if not a or not b:
        return 0.0

    def weights(tokens: list[str]) -> dict[str, int]:
        w = dict.fromkeys(tokens, 1)
        w[tokens[-1]] = 2
        return w

    wa, wb = weights(a), weights(b)
    shared = sum(min(wa[t], wb[t]) for t in wa.keys() & wb.keys())
    return shared / min(sum(wa.values()), sum(wb.values()))


# ------------------------------------------------------------------------ inputs


def _alias_errors(item: dict[str, Any], where: str) -> list[str]:
    aliases = item.get("aliases")
    if aliases is None:
        return []
    if not isinstance(aliases, list) or not all(
            isinstance(a, str) and a.strip() for a in aliases):
        return [f"{where}: `aliases` must be a list of non-empty names"]
    return []


def load_gt(path: Path) -> dict[str, Any]:
    """Read and validate gt.yaml (unknown keys are errors: they are usually typos)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise GTError(f"{path}: the top level must be a mapping")
    errors = [f"unknown top-level key {k!r}" for k in sorted(set(data) - TOP_KEYS)]
    for k, item in enumerate(data.get("changes") or [], 1):
        where = f"changes[{k}]"
        if not isinstance(item, dict) or not str(item.get("object") or "").strip():
            errors.append(f"{where}: needs a non-empty `object`")
            continue
        errors += [f"{where}: unknown key {x!r}" for x in sorted(set(item) - CHANGE_KEYS)]
        errors += _alias_errors(item, where)
        if item.get("type") not in TYPES:
            errors.append(f"{where}: `type` must be one of {', '.join(TYPES)}")
        for key in ("displacement_m", "rotation_deg"):
            if item.get(key) is not None:
                if not isinstance(item[key], int | float) or isinstance(item[key], bool):
                    errors.append(f"{where}: `{key}` must be a number")
                elif item.get("type") != "moved":
                    errors.append(f"{where}: `{key}` is for moved objects only")
        size = item.get("size_m")
        if size is not None and not (isinstance(size, list) and len(size) == 3 and all(
                isinstance(v, int | float) and v > 0 for v in size)):
            errors.append(f"{where}: `size_m` must be three positive numbers")
    for k, item in enumerate(data.get("unchanged_but_only_in_one_video") or [], 1):
        where = f"unchanged_but_only_in_one_video[{k}]"
        if not isinstance(item, dict) or not str(item.get("object") or "").strip():
            errors.append(f"{where}: needs a non-empty `object`")
            continue
        errors += [f"{where}: unknown key {x!r}" for x in sorted(set(item) - UNCHANGED_KEYS)]
        errors += _alias_errors(item, where)
        if item.get("seen_in") not in ("before", "after"):
            errors.append(f"{where}: `seen_in` must be before or after")
    if errors:
        raise GTError(f"{path}:\n  " + "\n  ".join(errors))
    data.setdefault("scene", path.parent.name)
    return data


# ----------------------------------------------------------------------- scoring


def summary_of(change: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    out = {"id": change["id"], "type": change.get("type"), "label": change.get("label"),
           "confidence": change.get("confidence")}
    return out if score is None else {**out, "label_score": round(score, 3)}


def name_tokens(item: dict[str, Any]) -> list[list[str]]:
    """Token lists of a GT object's name and of each of its aliases."""
    return [label_tokens(n) for n in [item["object"], *(item.get("aliases") or [])]]


def names_score(names: list[list[str]], tokens: list[str]) -> float:
    """Best `label_score` of a prediction's tokens over the GT names."""
    return max((label_score(n, tokens) for n in names), default=0.0)


def score(gt: dict[str, Any], changes: list[dict[str, Any]], threshold: float = 0.6,
          scale: float = 1.0) -> dict[str, Any]:
    """Match predictions to the GT (rules: module docstring); returns the full report."""
    tokens = {c["id"]: label_tokens(c.get("label")) for c in changes}
    used: set[int] = set()

    def best(obj: list[list[str]], types: set[str]) -> tuple[dict[str, Any], float] | None:
        hits = [(c, names_score(obj, tokens[c["id"]])) for c in changes
                if c["id"] not in used and c.get("type") in types]
        hits = [(c, s) for c, s in hits if s >= threshold]
        return max(hits, key=lambda h: (h[0].get("confidence") or 0.0, h[1]), default=None)

    gt_changes = gt.get("changes") or []
    hits: list[list[tuple[dict[str, Any], float]]] = [[] for _ in gt_changes]
    for k, item in enumerate(gt_changes):  # pass 1: same type
        hit = best(name_tokens(item), {item["type"]})
        if hit:
            hits[k] = [hit]
            used.add(hit[0]["id"])
    for k, item in enumerate(gt_changes):  # pass 2: a moved object as removed + added
        if not hits[k] and item["type"] == "moved":
            obj = name_tokens(item)
            for kind in ("removed", "added"):
                hit = best(obj, {kind})
                if hit:
                    hits[k].append(hit)
                    used.add(hit[0]["id"])

    rows = []
    for k, (item, found) in enumerate(zip(gt_changes, hits), 1):
        kinds = [c.get("type") for c, _ in found]
        status = ("found" if kinds == [item["type"]] else "type wrong"
                  if kinds == ["removed", "added"] else "partial" if found else "missed")
        row: dict[str, Any] = {"gt": k, "object": item["object"], "type": item["type"],
                               "status": status, "type_correct": status == "found",
                               "predictions": [summary_of(c, s) for c, s in found]}
        if item["type"] == "moved":
            row.update(displacement(item, found, scale))
        rows.append(row)

    unchanged = []
    for item in gt.get("unchanged_but_only_in_one_video") or []:
        obj = name_tokens(item)
        found = [(c, names_score(obj, tokens[c["id"]])) for c in changes if c["id"] not in used]
        found = [(c, s) for c, s in found if s >= threshold]
        used.update(c["id"] for c, _ in found)
        wrong = [summary_of(c, s) for c, s in found if c.get("type") != "unverified"]
        unchanged.append({"object": item["object"], "seen_in": item["seen_in"],
                          "correct": not wrong, "reported_as_change": wrong,
                          "unverified": [summary_of(c, s) for c, s in found
                                         if c.get("type") == "unverified"]})

    extra = [summary_of(c) for c in changes if c["id"] not in used]
    n_found = sum(r["status"] in ("found", "type wrong") for r in rows)
    return {
        "scene": gt.get("scene"), "threshold": threshold, "scale": scale, "gt": rows,
        "unchanged": unchanged,
        "extra": [e for e in extra if e["type"] != "unverified"],
        "extra_unverified": [e for e in extra if e["type"] == "unverified"],
        "counts": {"gt": len(rows), "found": n_found,
                   "type_correct": sum(r["type_correct"] for r in rows),
                   "missed": len(rows) - n_found,
                   "false_positives": sum(e["type"] != "unverified" for e in extra),
                   "unchanged_correct": sum(u["correct"] for u in unchanged),
                   "unchanged": len(unchanged)},
    }


def displacement(item: dict[str, Any], found: list[tuple[dict[str, Any], float]],
                 scale: float) -> dict[str, Any]:
    """Predicted displacement (and rotation if reliable) of a moved GT object, with errors."""
    pred = rot = None
    if len(found) == 1 and found[0][0].get("type") == "moved":
        transform = found[0][0].get("transform") or {}
        if transform.get("translation_m") is not None:
            pred = scale * float(transform["translation_m"])
        if transform.get("rotation_reliable") and transform.get("rotation_deg") is not None:
            rot = float(transform["rotation_deg"])
    elif len(found) == 2:
        a = (found[0][0].get("before") or {}).get("centroid")
        b = (found[1][0].get("after") or {}).get("centroid")
        if a and b:
            pred = scale * math.dist(a, b)
    gt_d, gt_r = item.get("displacement_m"), item.get("rotation_deg")
    gt_r = None if gt_r is None else min(abs(gt_r) % 360, 360 - abs(gt_r) % 360)
    return {"displacement_gt_m": gt_d, "displacement_pred_m": _round(pred),
            "displacement_err_m": _round(pred - gt_d) if None not in (pred, gt_d) else None,
            "rotation_gt_deg": gt_r, "rotation_pred_deg": _round(rot, 1),
            "rotation_err_deg": _round(abs(rot - gt_r), 1) if None not in (rot, gt_r) else None}


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


# ------------------------------------------------------------------------ output


def markdown(report: dict[str, Any], out_dir: Path) -> str:
    def num(v: Any, fmt: str = ".2f") -> str:
        return "-" if v is None else format(v, fmt)

    def preds(row: dict[str, Any]) -> tuple[str, str, str]:
        p = row["predictions"]
        return (" + ".join(f"#{x['id']} {x['type']}" for x in p) or "-",
                " / ".join(str(x["label"]) for x in p) or "-",
                " / ".join(num(x["confidence"]) for x in p) or "-")

    c = report["counts"]
    lines = [f"## GT score: {report['scene']} ({out_dir})", "",
             "| GT | object | type | result | prediction | predicted label | conf "
             "| displacement GT / pred / err (m) | rotation GT / pred / err (deg) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in report["gt"]:
        disp = rot = ""
        if row["type"] == "moved":
            disp = " / ".join(num(row[f"displacement_{k}_m"]) for k in ("gt", "pred", "err"))
            rot = " / ".join(num(row[f"rotation_{k}_deg"], ".0f") for k in ("gt", "pred", "err"))
        lines.append(f"| {row['gt']} | {row['object']} | {row['type']} | {row['status']} | "
                     + " | ".join(preds(row)) + f" | {disp} | {rot} |")
    lines += ["", f"Found {c['found']}/{c['gt']} (type correct {c['type_correct']}), missed "
              f"{c['missed']}, false positives {c['false_positives']}"
              + (f", unchanged objects correct {c['unchanged_correct']}/{c['unchanged']}"
                 if c["unchanged"] else "")
              + (f" (distances x{report['scale']:g})" if report["scale"] != 1 else "") + "."]
    if report["extra"]:
        lines += ["", "### Extra predictions (false positives)", "",
                  "| id | type | label | conf |", "|---|---|---|---|"]
        lines += [f"| {e['id']} | {e['type']} | {e['label']} | {num(e['confidence'])} |"
                  for e in report["extra"]]
    if report["extra_unverified"]:
        lines += ["", "Unverified, not in the GT: " + ", ".join(
            f"#{e['id']} {e['label']} ({num(e['confidence'])})"
            for e in report["extra_unverified"])]
    if report["unchanged"]:
        lines += ["", "### Objects seen in one video only (must not be reported as changed)",
                  "", "| object | seen in | result |", "|---|---|---|"]
        for u in report["unchanged"]:
            if u["reported_as_change"]:
                result = "WRONG: reported as " + ", ".join(
                    f"{x['type']} #{x['id']} ({x['label']})" for x in u["reported_as_change"])
            elif u["unverified"]:
                result = "correct: unverified " + ", ".join(f"#{x['id']}" for x in u["unverified"])
            else:
                result = "correct: not reported"
            lines.append(f"| {u['object']} | {u['seen_in']} | {result} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="pipeline output directory (changes.json)")
    parser.add_argument("gt", type=Path, help="ground truth gt.yaml")
    parser.add_argument("--json", action="store_true", help="also write <out_dir>/gt_score.json")
    parser.add_argument("--threshold", type=float, default=0.6, help="label match (0.6)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply predicted distances (metric scale from lidar_check.py)")
    args = parser.parse_args(argv)
    changes_path = args.out_dir / "changes.json"
    if not changes_path.is_file():
        parser.error(f"{changes_path} not found")
    try:
        gt = load_gt(args.gt)
    except (GTError, yaml.YAMLError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    changes = json.loads(changes_path.read_text()).get("changes") or []
    report = score(gt, changes, args.threshold, args.scale)
    print(markdown(report, args.out_dir))
    if args.json:
        path = args.out_dir / "gt_score.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
