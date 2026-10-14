"""Screenshots and GIFs of the RoomDiff web viewer, taken in headless Chromium (Playwright).

Usage: python scripts/shoot_viewer.py <run_dir or viewer_dir> --out docs/img/viewer/<name>
           [--url URL] [--width 1400 --height 850] [--scale 1.5] [--changes all|1,3]
           [--gif none|orbit|blink|both] [--lossless]

Serves the viewer folder (`<run_dir>/viewer` if it exists, else the folder itself) with
`python -m http.server` on a free local port, unless `--url` names a running viewer, drives
the page through the viewer's automation hook `window.roomdiff` (viewer/app.js) and writes
  <name>_viewer.png      the whole page in the home view,
  <name>_scene.png       the 3D view only, same camera,
  <name>_change<id>.png  the whole page with change <id> selected (close-up and crops),
  <name>_orbit.gif       one turn around the scene in 10 degree steps (3D view only),
  <name>_blink.gif       before and after clouds alternating, boxes shown (3D view only).
PNGs have 256 colours (about 3x smaller than full colour and hardly different; --lossless
keeps full colour). GIFs are 800 px wide, 100 ms per frame and loop; their palette shrinks
until each file is at most 3 MB. WebGL runs on SwiftShader, so no GPU is needed, and three.js
is vendored with the viewer (viewer/vendor/), so no internet connection is needed either.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Browser, ConsoleMessage, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

CHROMIUM_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
READY_MS = 60_000  # the viewer must be ready within this time
SETTLE_MS = 300  # wait after a camera jump before rendering and shooting
READY_JS = ("() => window.roomdiff?.ready === true"
            " || !!document.querySelector('#status.error:not([hidden])')")
RENDER_JS = ("() => { window.roomdiff.render(); return new Promise((done) =>"
             " requestAnimationFrame(() => requestAnimationFrame(done))); }")
CROPS_JS = ("() => [...document.querySelectorAll('.change.selected .crops img')]"
            ".every((img) => img.complete)")
SCROLL_JS = "() => document.querySelector('.change.selected')?.scrollIntoView({ block: 'nearest' })"
# 3D-view shots: no overlays, no frame (its 1 px border shows at fractional scale factors)
HIDE_OVERLAYS = ("#hint, #status { visibility: hidden !important; }"
                 " #view { border-color: transparent !important; border-radius: 0 !important; }")
GIF_WIDTH, GIF_FRAME_MS, GIF_MAX_BYTES = 800, 100, 3 * 2**20
GIF_COLORS = (256, 128, 64, 32)  # palette sizes tried until the GIF is small enough
ORBIT_FRAMES, ORBIT_STEP_DEG = 36, 10
BLINK_FRAMES, BLINK_HOLD = 8, 6  # before / after alternating, each held 6 x 100 ms


def fail(message: str, problems: list[str]) -> NoReturn:
    raise SystemExit("\n  ".join([f"[shoot] error: {message}", *problems]))


def viewer_folder(path: Path) -> Path:
    folder = path / "viewer" if (path / "viewer").is_dir() else path
    if not (folder / "index.html").is_file():
        raise SystemExit(f"[shoot] error: no index.html in {folder} (give a run or viewer folder)")
    return folder


@contextlib.contextmanager
def serve(folder: Path) -> Iterator[str]:
    """Serve `folder` with `python -m http.server` on a free local port; yields its URL."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while server.poll() is None and time.monotonic() < deadline:
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 1):
                break  # the server accepts connections
            time.sleep(0.1)
        else:
            raise SystemExit(f"[shoot] error: could not serve {folder}")
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


def open_viewer(browser: Browser, url: str, args: argparse.Namespace) -> tuple[Page, list[str]]:
    """Load the viewer and wait for `window.roomdiff.ready`; exits on the viewer's error box or
    a timeout. Returns the page and a live list of console errors and uncaught exceptions."""
    page = browser.new_page(viewport={"width": args.width, "height": args.height},
                            device_scale_factor=args.scale)
    problems: list[str] = []

    def on_console(message: ConsoleMessage) -> None:
        if message.type == "error":
            problems.append(f"console error: {message.text} ({message.location['url']})")

    page.on("console", on_console)
    page.on("pageerror", lambda error: problems.append(f"page error: {error}"))
    response = page.goto(url, wait_until="commit", timeout=READY_MS)
    if response is not None and not response.ok:
        fail(f"HTTP {response.status} for {url}", problems)
    try:
        page.wait_for_function(READY_JS, timeout=READY_MS, polling=200)
    except PlaywrightTimeout:
        fail(f"window.roomdiff.ready is not true after {READY_MS // 1000} s ({url})", problems)
    error_box = page.locator("#status.error")
    if error_box.is_visible():
        fail(f"the viewer shows an error: {error_box.inner_text()}", problems)
    page.evaluate("() => window.roomdiff.pauseLoop()")  # only render() draws from now on
    return page, problems


def settle(page: Page, ms: int = SETTLE_MS) -> None:
    """Wait `ms`, render the current camera and let two animation frames pass."""
    page.wait_for_timeout(ms)
    page.evaluate(RENDER_JS)


def canvas_png(page: Page) -> bytes:
    return page.locator("#canvas").screenshot(style=HIDE_OVERLAYS)


def gif_frame(page: Page, caption: str = "") -> Image.Image:
    """The 3D view scaled to GIF_WIDTH, with an optional caption in the top left corner."""
    image = Image.open(io.BytesIO(canvas_png(page))).convert("RGB")
    image = image.resize((GIF_WIDTH, round(image.height * GIF_WIDTH / image.width)),
                         Image.Resampling.LANCZOS)
    if caption:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=26)
        left, top, right, bottom = draw.textbbox((20, 14), caption, font=font)
        draw.rounded_rectangle((left - 10, top - 7, right + 10, bottom + 7), 6, fill=(0, 0, 0))
        draw.text((20, 14), caption, font=font, fill=(255, 255, 255))
    return image


def write_gif(path: Path, frames: list[Image.Image]) -> Path:
    """Looping GIF whose frames share one palette, without dithering (steady colours, good
    compression); the palette shrinks until the file is at most GIF_MAX_BYTES."""
    step = max(1, len(frames) // 8)
    sample = Image.fromarray(np.vstack([np.asarray(frame) for frame in frames[::step]]))
    for colors in GIF_COLORS:
        palette = sample.quantize(colors, method=Image.Quantize.MEDIANCUT)
        quantized = [np.asarray(frame.quantize(palette=palette, dither=Image.Dither.NONE)
                                .convert("RGB")) for frame in frames]
        data = iio.imwrite("<bytes>", quantized, extension=".gif", duration=GIF_FRAME_MS, loop=0)
        if len(data) <= GIF_MAX_BYTES:
            break
    else:
        print(f"[shoot] warning: {path.name}: {len(data) / 2**20:.1f} MB with {colors} colours")
    path.write_bytes(data)
    return path


def pick_changes(items: list[dict], spec: str) -> list[dict]:
    """The changes named by `spec` ("all" or change ids such as "1,3"), hidden ones skipped."""
    shown = {str(item["id"]): item for item in items if not item["hidden"]}
    if spec == "all":
        return list(shown.values())
    ids = [part.strip() for part in spec.split(",") if part.strip()]
    if unknown := [i for i in ids if i not in shown]:
        raise SystemExit(f"[shoot] error: no shown change with id {', '.join(unknown)} "
                         f"(ids: {', '.join(shown) or 'none'})")
    return [shown[i] for i in ids]


def shoot(page: Page, out: Path, args: argparse.Namespace) -> list[Path]:
    written: list[Path] = []

    def png(data: bytes, suffix: str) -> None:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        if not args.lossless:
            image = image.quantize(256, method=Image.Quantize.FASTOCTREE)
        written.append(Path(f"{out}_{suffix}.png"))
        image.save(written[-1], optimize=True)

    changes = pick_changes(page.evaluate("() => window.roomdiff.items()"), args.changes)
    page.evaluate("() => window.roomdiff.jumpHome()")
    settle(page)
    png(page.screenshot(full_page=True), "viewer")
    png(canvas_png(page), "scene")
    for item in changes:
        page.evaluate("(index) => window.roomdiff.jumpTo(index)", item["index"])
        page.wait_for_function(CROPS_JS, timeout=10_000)
        page.evaluate(SCROLL_JS)  # again: the entry grew when its crops loaded
        settle(page)
        png(page.screenshot(full_page=True), f"change{item['id']}")
    page.evaluate("() => window.roomdiff.jumpHome()")
    if args.gif in ("orbit", "both"):
        frames = []
        for _ in range(ORBIT_FRAMES):
            settle(page, 0)
            frames.append(gif_frame(page))
            page.evaluate("(deg) => window.roomdiff.orbit(deg)", ORBIT_STEP_DEG)
        page.evaluate("() => window.roomdiff.jumpHome()")
        written.append(write_gif(Path(f"{out}_orbit.gif"), frames))
    if args.gif in ("blink", "both"):
        shots = {}
        for side in ("before", "after"):  # true colours: tint off
            page.evaluate("(side) => { document.getElementById('tint').checked = false;"
                          " window.roomdiff.setClouds(side); }", side)
            settle(page)
            shots[side] = gif_frame(page, caption=side)
        page.evaluate("() => window.roomdiff.setClouds('both')")
        sides = ("before", "after") * (BLINK_FRAMES // 2)
        written.append(write_gif(Path(f"{out}_blink.gif"),
                                 [shots[side] for side in sides for _ in range(BLINK_HOLD)]))
    return written


def main(argv: list[str] | None = None) -> None:
    intro, _, details = __doc__.split("\n\n", 2)
    parser = argparse.ArgumentParser(description=intro, epilog=details,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, nargs="?",
                        help="run folder (with viewer/) or viewer folder; not needed with --url")
    parser.add_argument("--out", required=True,
                        help="output prefix, e.g. docs/img/viewer/living_room_1")
    parser.add_argument("--url", help="shoot this viewer URL instead of serving SOURCE")
    parser.add_argument("--width", type=int, default=1400, help="viewport width in CSS px (1400)")
    parser.add_argument("--height", type=int, default=850, help="viewport height in CSS px (850)")
    parser.add_argument("--scale", type=float, default=1.5, help="device scale factor (1.5)")
    parser.add_argument("--changes", default="all", help="'all' (default) or change ids, e.g. 1,3")
    parser.add_argument("--gif", choices=("none", "orbit", "blink", "both"), default="both",
                        help="GIFs to make (default both)")
    parser.add_argument("--lossless", action="store_true", help="full-colour PNGs")
    args = parser.parse_args(argv)
    if args.source is None and not args.url:
        parser.error("give the run or viewer folder, or --url")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # stop the server on a timeout too
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        url = args.url or stack.enter_context(serve(viewer_folder(args.source)))
        browser = stack.enter_context(sync_playwright()).chromium.launch(args=CHROMIUM_ARGS)
        stack.callback(browser.close)
        page, problems = open_viewer(browser, url, args)
        written = shoot(page, out, args)
    for problem in problems:
        print(f"[shoot] warning: {problem}")
    for path in written:
        print(f"{path}  {path.stat().st_size / 2**20:.2f} MB")
    print(f"total {sum(path.stat().st_size for path in written) / 2**20:.2f} MB")


if __name__ == "__main__":
    main()
