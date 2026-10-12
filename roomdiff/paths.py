"""Output directory layout shared by all pipeline stages.

```
<out>/keyframes/before/*.jpg     <out>/recon/          <out>/changes.json
<out>/keyframes/after/*.jpg      <out>/visibility/     <out>/report.md
<out>/keyframes/index.json       <out>/instances/      <out>/viewer/
```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RECORDINGS = ("before", "after")


@dataclass
class OutputPaths:
    """All output locations below one run directory `root`."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def keyframes(self) -> Path:
        return self.root / "keyframes"

    def keyframes_of(self, recording: str) -> Path:
        """Keyframe JPEG directory of one recording ("before" or "after")."""
        if recording not in RECORDINGS:
            raise ValueError(f"recording must be one of {RECORDINGS}, got {recording!r}")
        return self.keyframes / recording

    @property
    def keyframes_index(self) -> Path:
        return self.keyframes / "index.json"

    @property
    def recon(self) -> Path:
        return self.root / "recon"

    @property
    def visibility(self) -> Path:
        return self.root / "visibility"

    @property
    def instances(self) -> Path:
        return self.root / "instances"

    @property
    def changes_json(self) -> Path:
        return self.root / "changes.json"

    @property
    def report_md(self) -> Path:
        return self.root / "report.md"

    @property
    def viewer(self) -> Path:
        return self.root / "viewer"
