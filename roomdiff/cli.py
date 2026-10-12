"""Command-line interface: `python -m roomdiff.cli <command> --help`."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer

from roomdiff.config import Config, ConfigError, load_config

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
def instances(out: OutOpt, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Object instances of the change blobs, SAM 2.1 + DINOv3 (<out>/instances/)."""
    _instances_stage(out, _load_config(config, set_))


@app.command()
def changes(out: OutOpt, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Match instances across recordings and write <out>/changes.json and <out>/crops/."""
    _changes_stage(out, _load_config(config, set_))


@app.command()
def run(
    before: BeforeOpt,
    after: AfterOpt,
    out: OutOpt,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Run ingest -> recon -> visibility -> instances -> changes -> viz."""
    from roomdiff.ingest import ingest_pair
    from roomdiff.recon import run_recon

    cfg = _load_config(config, set_)
    stages = {
        "ingest": lambda: ingest_pair(before, after, out, cfg),
        "recon": lambda: run_recon(out, cfg),
        "visibility": lambda: _visibility_stage(out, cfg),
        "instances": lambda: _instances_stage(out, cfg),
        "changes": lambda: _changes_stage(out, cfg),
        "viz": lambda: _viz_stage(out),
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


@app.command("eval")
def eval_(out: OutOpt) -> None:
    """Evaluate a run against ground truth."""
    _not_implemented("eval")


def _visibility_stage(out: Path, cfg: Config) -> None:
    from roomdiff.visibility import run_visibility  # torch / scipy only when needed
    from roomdiff.viz_rerun import log_visibility

    run_visibility(out, cfg)
    log_visibility(out)


def _instances_stage(out: Path, cfg: Config) -> None:
    from roomdiff.instances import run_instances  # models load only when needed

    run_instances(out, cfg)


def _changes_stage(out: Path, cfg: Config) -> None:
    from roomdiff.changes import run_changes
    from roomdiff.viz_rerun import log_changes

    run_changes(out, cfg)
    log_changes(out)


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
