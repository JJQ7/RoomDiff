"""M5 report: `<out>/report.md` from `<out>/changes.json` (after the semantics stage).

Title, summary paragraph, counts table, one section per change (type badge, label,
confidence, description, evidence line, crops as relative markdown images), a collapsible
list of the rejected candidates with their `evidence.reason_class` as a tag and the VLM's
reason, and a coverage note from `recon/meta.json` (share of each recording's points the
other one could observe).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roomdiff.changes import CHANGE_TYPES
from roomdiff.paths import OutputPaths

PLACES_NOTE = ("- Places (left / right, near / far) are as seen along the average camera "
               "direction; distances are in approximate metres from the reconstruction.")
CROP_TITLES = {
    "before": "before",
    "after": "after",
    "before_place": "after video, old place",
    "after_place": "before video, new place",
    "before_part": "before, merged part",
    "after_part": "after, merged part",
}


def _fmt(value: Any, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def evidence_line(change: Mapping[str, Any]) -> str:
    """Short human-readable evidence summary of one change."""
    ev, kind = change.get("evidence") or {}, change["type"]
    parts = []
    if kind == "moved":
        parts.append(f"old place seen empty from {ev.get('observing_views_before_place', 0)} "
                     f"views of the after video, new place from "
                     f"{ev.get('observing_views_after_place', 0)} views of the before video")
        if ev.get("appearance_similarity") is not None:
            parts.append(f"appearance similarity {_fmt(ev['appearance_similarity'])}")
        tf = change.get("transform") or {}
        if tf.get("translation_m") is not None:
            parts.append(f"displacement {tf['translation_m']:.2f} m")
        if tf.get("rotation_deg") is not None:
            unreliable = "" if tf.get("rotation_reliable", True) else " (unreliable)"
            parts.append(f"rotation{unreliable} {tf['rotation_deg']:.0f} deg")
        if change.get("motion_kind"):
            parts.append(f"motion {change['motion_kind']}")
    elif kind in ("removed", "added"):
        other = "after" if kind == "removed" else "before"
        rays = f" ({ev['free_rays']} free-space rays)" if ev.get("free_rays") is not None else ""
        parts.append(f"place seen empty from {ev.get('observing_views', 0)} views of the "
                     f"{other} video{rays}")
    else:
        other = "after" if ev.get("seen_in") == "before" else "before"
        parts.append(f"seen only in the {ev.get('seen_in')} video; the {other} video never "
                     "observed the place")
    app = ev.get("appearance")
    if app:
        parts.append(f"appearance channel: score {_fmt(app['mean_score'])} (feature "
                     f"{_fmt(app['feat_term'])}, depth {_fmt(app['geom_term'])}) in "
                     f"{app['frames_supporting']} keyframes")
    if change.get("replaced_by") is not None:
        parts.append(f"same place as added change {change['replaced_by']}")
    if change.get("replaces") is not None:
        parts.append(f"same place as removed change {change['replaces']}")
    if ev.get("vlm_replace_answer"):
        parts.append(f"in-place check: {ev['vlm_replace_answer']}")
    if len(ev.get("merged_from", [])) > 1:  # re-paired, or parts of one object
        parts.append("merged from candidates " + ", ".join(map(str, ev["merged_from"])))
    if ev.get("mask_score") is not None:
        parts.append(f"mask score {_fmt(ev['mask_score'])}")
    if ev.get("vlm_verdict"):
        reason = f" ({ev['vlm_reason']})" if ev.get("vlm_reason") else ""
        parts.append(f"VLM {ev['vlm_verdict']}{reason}")
    recheck = ev.get("recheck") or {}
    if recheck.get("verdict") not in (None, "skipped"):  # optional SAM 3 stage
        parts.append(f"SAM 3 re-check {recheck['verdict']}: {recheck['reason']}")
    return "; ".join(parts)


def crop_table(crops: Mapping[str, str | None] | None, cid: Any) -> list[str]:
    """Markdown table with one column per existing crop."""
    items = [(key, path) for key, path in (crops or {}).items() if path]
    if not items:
        return []
    order = list(CROP_TITLES)
    items.sort(key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order))
    head = "| " + " | ".join(CROP_TITLES.get(k, k) for k, _ in items) + " |"
    rule = "|" + "---|" * len(items)
    row = "| " + " | ".join(f"![{cid} {k}]({p})" for k, p in items) + " |"
    return [head, rule, row]


def change_section(change: Mapping[str, Any]) -> list[str]:
    cid, kind = change["id"], change["type"]
    ev = change.get("evidence") or {}
    facts = [f"**confidence** {change['confidence']:.2f}"]
    if ev.get("vlm_verdict"):
        facts.append(f"**VLM verdict** {ev['vlm_verdict']}")
    if change.get("motion_kind"):
        facts.append(f"**motion** {change['motion_kind']}")
    if change.get("attributes"):
        facts.append(f"**appearance** {change['attributes']}")
    lines = [f"### {cid}. `{kind.upper()}` {change.get('label', 'object')}", "",
             " · ".join(facts), "", change.get("description", ""), "",
             f"*Evidence:* {evidence_line(change)}", ""]
    table = crop_table(change.get("crops"), cid)
    return lines + (table + [""] if table else [])


def coverage_lines(paths: OutputPaths) -> list[str]:
    """Coverage bullet points from `recon/meta.json` (empty if unavailable)."""
    meta_file = paths.recon / "meta.json"
    if not meta_file.is_file():
        return []
    diag = json.loads(meta_file.read_text()).get("diagnostics", {})
    covis = diag.get("nn_covisible", {})
    lines = []
    for key, own, other in (("before_to_after", "before", "after"),
                            ("after_to_before", "after", "before")):
        if key in covis:
            lines.append(f"- {100 * covis[key]['fraction']:.0f}% of the {own} video's valid "
                         f"points lie where the {other} video could observe them (median "
                         f"distance there {100 * covis[key]['median_m']:.1f} cm).")
    if diag.get("verdict"):
        reason = f" ({diag['verdict_reason']})" if diag.get("verdict_reason") else ""
        lines.append(f"- Reconstruction check: {diag['verdict']}{reason}.")
    return lines


def render_report(doc: Mapping[str, Any], title: str, coverage: Sequence[str]) -> str:
    """Markdown text of the report for one changes document."""
    changes, rejected = doc.get("changes", []), doc.get("rejected", [])
    scene = doc.get("scene", {})
    counts = Counter(c["type"] for c in changes)
    lines = [f"# RoomDiff report: {title}", ""]
    videos = [f"{rec} video `{scene[rec]}`" for rec in ("before", "after") if rec in scene]
    if videos or scene.get("units"):
        lines += [", ".join(videos) + (f"; units: {scene['units']}" if scene.get("units")
                                       else ""), ""]
    lines += ["## Summary", "", doc.get("summary", ""), "",
              "| type | count |", "|---|---:|"]
    lines += [f"| {t} | {counts[t]} |" for t in CHANGE_TYPES]
    lines += [f"| rejected candidates | {len(rejected)} |", "", "## Changes", ""]
    if not changes:
        lines += ["No change was found.", ""]
    for change in changes:
        lines += change_section(change)
    if rejected:
        lines += [f"<details><summary>Rejected candidates ({len(rejected)})</summary>", ""]
        for c in rejected:
            ev = c.get("evidence") or {}
            crop = next((p for p in (c.get("crops") or {}).values() if p), None)
            link = f" ([crop]({crop}))" if crop else ""
            reason = ev.get("vlm_reason") or "no reason"
            if ev.get("recheck_verdict") == "contradicted":
                reason = f"SAM 3 re-check: {ev['recheck']['reason']} (VLM: {reason})"
            tag = f" `{ev['reason_class']}`" if ev.get("reason_class") else ""
            lines.append(f"- **{c['id']}** {c['type']} *{c.get('label', 'object')}*{tag}, "
                         f"confidence {c['confidence']:.2f}: {reason}{link}")
        lines += ["", "</details>", ""]
    lines += ["## Coverage", ""]
    lines += list(coverage) or ["- No reconstruction diagnostics available."]
    lines += [PLACES_NOTE, ""]
    return "\n".join(lines)


def write_report(out_dir: str | Path) -> Path:
    """Write `<out>/report.md` from `<out>/changes.json`; returns its path."""
    paths = OutputPaths(out_dir)
    doc = json.loads(paths.changes_json.read_text())
    text = render_report(doc, paths.root.resolve().name, coverage_lines(paths))
    paths.report_md.write_text(text)
    print(f"[report] {len(doc.get('changes', []))} changes, "
          f"{len(doc.get('rejected', []))} rejected -> {paths.report_md}")
    return paths.report_md
