"""Command-line interface: `python -m roomdiff.cli <command> --help`."""

from __future__ import annotations

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
def run(
    before: BeforeOpt,
    after: AfterOpt,
    out: OutOpt,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Run the full pipeline (ingest -> recon -> visibility -> instances -> report)."""
    _not_implemented("run")


@app.command()
def viz(out: OutOpt) -> None:
    """Open the results of a run in Rerun / build the web viewer assets."""
    _not_implemented("viz")


@app.command("eval")
def eval_(out: OutOpt) -> None:
    """Evaluate a run against ground truth."""
    _not_implemented("eval")


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
