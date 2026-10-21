"""Command-line interface: `python -m roomdiff.cli <command> --help`."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer

from roomdiff.config import Config, ConfigError, load_config

TARGET_POINTS = 400_000  # viewer cloud size (roomdiff.viewer_export.TARGET_POINTS)

app = typer.Typer(
    help="RoomDiff: what changed in the room between two phone videos?",
    no_args_is_help=True,
    add_completion=False,
)

BeforeOpt = Annotated[
    Path, typer.Option(exists=True, dir_okay=False, help="Video of the room before.")
]
AfterOpt = Annotated[
    Path, typer.Option(exists=True, dir_okay=False, help="Video of the room after.")
]
OutOpt = Annotated[Path, typer.Option(help="Output directory.")]
ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config", "-c", exists=True, dir_okay=False,
        help="YAML file overriding configs/default.yaml.",
    ),
]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override one value, e.g. ingest.num_keyframes=32."),
]


@app.command()
def ingest(
    before: BeforeOpt,
    after: AfterOpt,
    out: OutOpt,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Decode both videos and write sharp, upright keyframes to <out>/keyframes/."""
    from roomdiff.ingest import ingest_pair  # heavy imports (av, cv2) only when needed

    ingest_pair(before, after, out, _load_config(config, set_))


@app.command()
def recon(out: OutOpt, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Reconstruct both keyframe sets with Pi3X in one metric frame (<out>/recon/)."""
    from roomdiff.recon import run_recon  # heavy imports (torch, pi3) only when needed

    run_recon(out, _load_config(config, set_))


@app.command()
def visibility(out: OutOpt, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Voxel occupied / free / unknown grids and change candidates (<out>/visibility/)."""
    _visibility_stage(out, _load_config(config, set_))


@app.command()
def appearance(out: OutOpt, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Small / flat change candidates from reprojected DINOv3 features (<out>/appearance/)."""
    _appearance_stage(out, _load_config(config, set_))


@app.command()
def detect(out: OutOpt, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Detection channel: Qwen3-VL vocabulary, SAM 3 objects in every keyframe, before /
    after association (<out>/detection/; needs --set detection.enabled=true)."""
    _detect_stage(out, _load_config(config, set_))


@app.command()
def instances(out: OutOpt, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Object instances of the change blobs, SAM 2.1 + DINOv3 (<out>/instances/)."""
    _instances_stage(out, _load_config(config, set_))


@app.command()
def changes(out: OutOpt, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Match instances across recordings and write <out>/changes.json and <out>/crops/."""
    _changes_stage(out, _load_config(config, set_))


@app.command()
def relations(out: OutOpt, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Where each change sits (support, on / under / beside) into <out>/changes.json, no
    model; rewrites report.md after describe. `changes` and `describe` already do this."""
    from roomdiff.relations import run_relations

    run_relations(out, _load_config(config, set_))


@app.command()
def describe(out: OutOpt, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Name, verify (VLM judge) and describe the changes with Qwen3-VL; write report.md."""
    _describe_stage(out, _load_config(config, set_))


@app.command()
def recheck(out: OutOpt, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Optional: SAM 3 re-detects each change's label on the object and at its place in the
    other video; rewrites changes.json and report.md (needs --set recheck.enabled=true)."""
    _recheck_stage(out, _load_config(config, set_))


@app.command()
def run(
    before: BeforeOpt,
    after: AfterOpt,
    out: OutOpt,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Run ingest -> recon -> visibility -> appearance -> detect (if detection.enabled) ->
    instances -> changes -> describe -> (recheck, if recheck.enabled) -> viz -> viewer."""
    from roomdiff.ingest import ingest_pair
    from roomdiff.recon import run_recon

    cfg = _load_config(config, set_)
    optional = {"recheck": lambda: _recheck_stage(out, cfg)} if cfg.recheck.enabled else {}
    stages = {
        "ingest": lambda: ingest_pair(before, after, out, cfg),
        "recon": lambda: run_recon(out, cfg),
        "visibility": lambda: _visibility_stage(out, cfg),
        "appearance": lambda: _appearance_stage(out, cfg),
        "detect": lambda: _detect_stage(out, cfg),
        "instances": lambda: _instances_stage(out, cfg),
        "changes": lambda: _changes_stage(out, cfg),
        "describe": lambda: _describe_stage(out, cfg),
        **optional,
        "viz": lambda: _viz_stage(out),
        "viewer": lambda: _viewer_stage(out, TARGET_POINTS),
    }
    seconds = {}
    for name, stage in stages.items():
        start = time.perf_counter()
        stage()
        seconds[name] = time.perf_counter() - start
    typer.echo("[run] wall time: " + ", ".join(f"{k} {v:.1f} s" for k, v in seconds.items()))


@app.command()
def viz(out: OutOpt) -> None:
    """Write the Rerun recordings <out>/scene.rrd (everything) and <out>/recon/recon.rrd."""
    _viz_stage(out)


@app.command()
def viewer(
    out: OutOpt,
    target_points: Annotated[int, typer.Option(help="Max points per cloud.")] = TARGET_POINTS,
) -> None:
    """Export the static web viewer to <out>/viewer/ (serve it with python -m http.server)."""
    _viewer_stage(out, target_points)


@app.command("eval")
def eval_(
    pairs: Annotated[
        str | None, typer.Option(help="Comma-separated SceneDiff pair folder names.")
    ] = None,
    split: Annotated[
        str | None, typer.Option(help="Take the pairs of this split instead: val or test.")
    ] = None,
    subset: Annotated[
        str, typer.Option("--set", help="Subset of --split: varied, kitchen or all.")
    ] = "varied",
    limit: Annotated[int | None, typer.Option(help="Only the first N pairs.")] = None,
    out_root: Annotated[
        Path, typer.Option(help="Runs, predictions and results are written here.")
    ] = Path("outputs/scenediff_eval"),
    skip_run: Annotated[
        bool, typer.Option("--skip-run", help="Only reuse <out-root>/runs/<pair>/ outputs.")
    ] = False,
    config: ConfigOpt = None,
    override: Annotated[
        list[str] | None,
        typer.Option("--override", "-o",
                     help="Config override, also passed to every run, e.g. eval.time_tol_s=0.2."),
    ] = None,
) -> None:
    """SceneDiff benchmark: run the pipeline with 1 s keyframes on each pair, write
    object_masks.pkl and score it with the official evaluator (<out-root>/pred/summary.json)."""
    from roomdiff.eval_scenediff import benchmark, data_root, select_pairs

    cfg = _load_config(config, override)
    if pairs:
        names = [p.strip() for p in pairs.split(",") if p.strip()]
    elif split in ("val", "test") and subset in ("varied", "kitchen", "all"):
        names = select_pairs(data_root(cfg), split, subset)
    else:
        raise typer.BadParameter("give --pairs, or --split val|test with --set varied|kitchen|all")
    names = names[:limit] if limit else names
    missing = [p for p in names if not (data_root(cfg) / "data" / p).is_dir()]
    if not names or missing:
        raise typer.BadParameter(f"no such SceneDiff pair folder(s): {missing or names}")
    benchmark(names, out_root, cfg, skip_run, config, override or [])


def _visibility_stage(out: Path, cfg: Config) -> None:
    from roomdiff.visibility import run_visibility  # torch / scipy only when needed
    from roomdiff.viz_rerun import log_visibility

    run_visibility(out, cfg)
    log_visibility(out)


def _appearance_stage(out: Path, cfg: Config) -> None:
    from roomdiff.appearance import run_appearance  # models load only when needed

    run_appearance(out, cfg)


def _detect_stage(out: Path, cfg: Config) -> None:
    from roomdiff.detection import run_detection  # models load only when needed

    run_detection(out, cfg)


def _instances_stage(out: Path, cfg: Config) -> None:
    from roomdiff.instances import run_instances  # models load only when needed

    run_instances(out, cfg)


def _changes_stage(out: Path, cfg: Config) -> None:
    from roomdiff.changes import run_changes
    from roomdiff.viz_rerun import log_changes

    run_changes(out, cfg)
    log_changes(out)


def _describe_stage(out: Path, cfg: Config) -> None:
    from roomdiff.report import write_report
    from roomdiff.semantics import run_semantics  # the VLM loads only when needed
    from roomdiff.viz_rerun import log_changes

    run_semantics(out, cfg)
    write_report(out)
    log_changes(out)


def _recheck_stage(out: Path, cfg: Config) -> None:
    from roomdiff.recheck import run_recheck  # SAM 3 loads only when needed
    from roomdiff.viz_rerun import log_changes

    if run_recheck(out, cfg) is not None:  # rewrites changes.json and report.md
        log_changes(out)


def _viewer_stage(out: Path, target_points: int) -> None:
    from roomdiff.viewer_export import export_viewer

    export_viewer(out, target_points)


def _viz_stage(out: Path) -> None:
    from roomdiff.viz_rerun import log_recon, log_scene

    log_scene(out)
    log_recon(out)  # reconstruction-only file, kept for compatibility (about 1 s)


def _load_config(config: Path | None, overrides: list[str] | None) -> Config:
    try:
        return load_config(override_file=config, overrides=overrides or [])
    except ConfigError as err:  # typo in --config / --set: report it without a traceback
        raise typer.BadParameter(str(err)) from None


def _not_implemented(command: str) -> None:
    typer.echo(f"roomdiff {command}: not implemented yet", err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
