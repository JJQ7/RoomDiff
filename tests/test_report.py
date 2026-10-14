"""Tests for the M5 markdown report (no models, no GPU)."""

import json

from roomdiff.report import evidence_line, render_report, write_report

DOC = {
    "scene": {"before": "v1.mp4", "after": "v2.mp4", "units": "metres (approx.)"},
    "summary": "2 changes: 1 moved, 0 added, 1 removed; 1 unverified. The ladder moved.",
    "changes": [
        {"id": 1, "type": "moved", "label": "wooden ladder", "attributes": "brown wood",
         "confidence": 0.77, "motion_kind": "rigid",
         "transform": {"translation_m": 0.42, "rotation_deg": 12.0},
         "evidence": {"observing_views_before_place": 4, "observing_views_after_place": 9,
                      "appearance_similarity": 0.7653, "mask_score": 0.8,
                      "vlm_verdict": "confirmed", "vlm_reason": "same ladder"},
         "description": "The wooden ladder moved 0.4 m.",
         "crops": {"before": "crops/1_before.jpg", "after": "crops/1_after.jpg",
                   "before_place": "crops/1_before_place.jpg", "after_place": None}},
        {"id": 3, "type": "removed", "label": "brown bag", "confidence": 0.92,
         "evidence": {"observing_views": 7, "free_rays": 120, "vlm_verdict": "confirmed",
                      "vlm_reason": "gone"},
         "description": "The brown bag was removed.", "crops": {"before": "crops/3_before.jpg"}},
        {"id": 7, "type": "unverified", "label": "dresser", "confidence": 0.31,
         "evidence": {"seen_in": "after", "vlm_verdict": "unclear", "vlm_reason": "dark"},
         "description": "A dresser was seen only after.", "crops": {}},
    ],
    "rejected": [
        {"id": 9, "type": "unverified", "label": "wall", "confidence": 0.41,
         "evidence": {"vlm_verdict": "rejected", "vlm_reason": "part of the wall"},
         "crops": {"after": "crops/9_after.jpg"}},
    ],
}


def test_render_report_sections():
    text = render_report(DOC, "scene_x", ["- 76% of the before video's points ..."])
    lines = text.splitlines()
    assert lines[0] == "# RoomDiff report: scene_x"
    assert "before video `v1.mp4`, after video `v2.mp4`; units: metres (approx.)" in text
    assert DOC["summary"] in text
    for row in ("| moved | 1 |", "| removed | 1 |", "| added | 0 |", "| unverified | 1 |",
                "| rejected candidates | 1 |"):
        assert row in lines
    assert "### 1. `MOVED` wooden ladder" in lines and "### 3. `REMOVED` brown bag" in lines
    assert text.index("### 1.") < text.index("### 3.") < text.index("### 7.")
    assert "**confidence** 0.77 · **VLM verdict** confirmed · **motion** rigid" in text
    # crops as relative markdown images, one column per existing crop
    assert "| before | after | after video, old place |" in lines
    assert "| ![1 before](crops/1_before.jpg) | ![1 after](crops/1_after.jpg) | " \
        "![1 before_place](crops/1_before_place.jpg) |" in lines
    assert "after_place" not in text.split("### 3.")[0].split("| before |")[1].split("\n")[0]
    # rejected list collapsible, with the reason and a crop link
    assert "<details><summary>Rejected candidates (1)</summary>" in text
    assert "- **9** unverified *wall*, confidence 0.41: part of the wall " \
        "([crop](crops/9_after.jpg))" in lines
    assert "## Coverage" in text and "- 76% of the before video's points ..." in lines


def test_evidence_line():
    moved = evidence_line(DOC["changes"][0])
    assert moved.startswith("old place seen empty from 4 views of the after video, new place "
                            "from 9 views of the before video; appearance similarity 0.77")
    assert "displacement 0.42 m; rotation 12 deg; motion rigid" in moved
    shaky = {**DOC["changes"][0], "transform": {"translation_m": 0.42, "rotation_deg": 14.8,
                                                "rotation_reliable": False}}
    assert "displacement 0.42 m; rotation (unreliable) 15 deg; motion rigid" in evidence_line(shaky)
    assert moved.endswith("VLM confirmed (same ladder)")
    removed = evidence_line(DOC["changes"][1])
    assert removed.startswith("place seen empty from 7 views of the after video (120 free-space")
    assert evidence_line(DOC["changes"][2]).startswith("seen only in the after video")


def test_write_report_with_coverage(tmp_path):
    (tmp_path / "changes.json").write_text(json.dumps(DOC))
    (tmp_path / "recon").mkdir()
    diag = {"nn_covisible": {"before_to_after": {"fraction": 0.758, "median_m": 0.0087},
                             "after_to_before": {"fraction": 0.464, "median_m": 0.0076}},
            "verdict": "alignment OK", "verdict_reason": ""}
    (tmp_path / "recon" / "meta.json").write_text(json.dumps({"diagnostics": diag}))
    path = write_report(tmp_path)
    assert path == tmp_path / "report.md"
    text = path.read_text()
    assert f"# RoomDiff report: {tmp_path.name}" in text
    assert "- 76% of the before video's valid points lie where the after video could observe " \
        "them (median distance there 0.9 cm)." in text
    assert "- 46% of the after video's valid points" in text
    assert "- Reconstruction check: alignment OK." in text


def test_coverage_note_on_the_alignment_gate(tmp_path):
    """The 3 cm share of the weaker direction (recomputed for an older recon/meta.json
    without `frac_lt_3cm_min`), and the M3 blobs demoted to unverified."""
    (tmp_path / "changes.json").write_text(json.dumps(DOC))
    (tmp_path / "recon").mkdir()
    covis = {"before_to_after": {"fraction": 0.278, "median_m": 0.094, "frac_lt_3cm": 0.278},
             "after_to_before": {"fraction": 0.177, "median_m": 0.074, "frac_lt_3cm": 0.293}}
    reason = "co-visible median > 2 cm or < 90% within 10 cm"
    (tmp_path / "recon" / "meta.json").write_text(json.dumps({"diagnostics": {
        "nn_covisible": covis, "verdict": "alignment WEAK", "verdict_reason": reason}}))
    (tmp_path / "visibility").mkdir()
    (tmp_path / "visibility" / "meta.json").write_text(json.dumps({
        "alignment_verdict": "alignment WEAK", "demoted": {"disappeared": 10, "appeared": 14},
        "demotion_reason": "weak_alignment"}))
    lines = write_report(tmp_path).read_text().splitlines()
    assert f"- Reconstruction check: alignment WEAK ({reason}); in the weaker direction 28% " \
        "of the co-visible points lie within 3 cm of the other video's points." in lines
    assert "- 24 geometry candidates (10 disappeared and 14 appeared) were set to unverified " \
        "because the two reconstructions disagree (alignment WEAK), so the other video " \
        "seeing those places empty is no evidence of a change." in lines
    (tmp_path / "recon" / "meta.json").write_text(json.dumps({"diagnostics": {
        "nn_covisible": covis, "verdict": "alignment FAIR", "verdict_reason": "few within 3 cm",
        "frac_lt_3cm_min": 0.774}}))
    (tmp_path / "visibility" / "meta.json").write_text(json.dumps({
        "alignment_verdict": "alignment FAIR", "demoted": {"disappeared": 0, "appeared": 0},
        "demotion_reason": None}))
    text = write_report(tmp_path).read_text()
    assert "- Reconstruction check: alignment FAIR (few within 3 cm); in the weaker direction " \
        "77% of the co-visible points" in text
    assert "set to unverified" not in text


def test_evidence_line_of_a_demoted_region():
    change = {**DOC["changes"][2], "evidence": {"seen_in": "before",
                                                "reason_class": "weak_alignment"}}
    assert evidence_line(change).startswith(
        "seen in the before video; the after video seemed to see the place empty, but the "
        "reconstructions are too poorly aligned to trust it (alignment WEAK)")
    never = {**DOC["changes"][2], "evidence": {"seen_in": "after",
                                               "reason_class": "never_observed"}}
    assert evidence_line(never).startswith("seen only in the after video; the before video "
                                           "never observed the place")


def test_report_without_changes_or_recon(tmp_path):
    doc = {"scene": {}, "changes": [], "rejected": [], "summary": "0 changes."}
    (tmp_path / "changes.json").write_text(json.dumps(doc))
    text = write_report(tmp_path).read_text()
    assert "No change was found." in text and "<details>" not in text
    assert "- No reconstruction diagnostics available." in text
