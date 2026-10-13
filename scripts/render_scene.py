"""Static images of a RoomDiff run for the README: the after point cloud with the change
boxes, the before / after clouds from the same camera and a before / after cross-fade GIF.

Usage:
    python scripts/render_scene.py outputs/scenediff_living_room_1 \
        --out docs/img/scenediff_living_room_1 [--width 1400] [--gif] [--view X,Y,Z]

Reads `<run>/viewer/{before,after}.ply` (x y z r g b), `<run>/viewer/scene.json`
(`up_axis`, `bbox_m`, `stats.voxel_m`, `name`) and `<run>/changes.json` (else the viewer's
copy) and writes
  <out>_overview.png      the after cloud with every change's oriented box: removed red,
                          added green, moved blue (before box dashed and thinner, arrow from
                          the before to the after centroid), unverified grey dashed; an
                          "<id> <label>" tag per change and a legend with the counts per type
  <out>_before.png, <out>_after.png   the same camera, clouds only
  <out>_beforeafter.gif   (--gif) cross-fade before -> after -> before, 900 px wide
Removed boxes sit on the after cloud where the object no longer is (the empty place). A PNG
over 600 KB is saved with a 256-colour palette (the overlay colours kept exact).

No GPU, browser or OpenGL (numpy, scipy, PIL): the points are projected with a pinhole
camera and splatted as squares the size of their voxel's projection (at least 3 px at
1400 px width) into a z-buffer (the nearest point wins), 1 px holes are closed and the
result is shaded with eye-dome lighting (a pixel lying behind its neighbours is darkened,
which outlines objects); dark clouds get a mild gamma. Boxes, arrows and tags are drawn on
a 2x layer, downsampled (anti-aliased) and laid over the points without a depth test, as
in the web viewer.

Camera: looks at the centre of `bbox_m` (in the frame where `up_axis` is +Y) from the
direction (-0.6, 0.75, -1.0), about 33 deg above the horizon, or from (-0.6, 0.75, 1.0) if
that view sees more of the points unoccluded (coarse z-buffer): the default would then
face the backs of the walls the videos filmed. The distance makes the bounding sphere of
`bbox_m` fill a 40 deg field of view; focal length, principal point and image height
(0.5-0.75 x the width) then fit the points inside `bbox_m` (0.5-99.5 percentiles) and all
change boxes with a 4 % margin. `--view X,Y,Z` sets the direction by hand.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from scipy.spatial import cKDTree

TYPES = ("removed", "added", "moved", "unverified")
TYPE_COLORS = {
    "removed": (229, 72, 77),  # #e5484d
    "added": (48, 164, 108),  # #30a46c
    "moved": (62, 142, 247),  # #3e8ef7
    "unverified": (139, 141, 152),  # #8b8d98
}
BACKGROUND = (20, 22, 26)  # #14161a
PANEL = (20, 22, 26, 228)  # tag and legend background
HALO = (10, 11, 13, 150)  # dark outline under the lines, for bright clouds
TEXT = (230, 231, 234)
VIEW_DIRECTION = (-0.6, 0.75, -1.0)  # eye offset from the look-at point, +Y up
FOV_DEG = 40.0  # the bounding sphere of bbox_m fills this field of view
MARGIN = 0.04  # of the image size, around the projected bbox_m
ASPECT_RANGE = (0.5, 0.75)  # image height / width
NEAR_M = 0.05
SUPERSAMPLE = 2  # overlay drawn at this scale, then downsampled
EDL_STRENGTH = 150.0
DARK_MEDIAN = 110  # clouds with a darker median luminance get a gamma
GIF_WIDTH = 900
GIF_DURATIONS_MS = (1000, 150, 150, 1000, 150, 150)  # before, 2 blends, after, 2 blends
MAX_PNG_BYTES = 600_000
MAX_GIF_BYTES = 3_000_000
FONT_FILES = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf")


# ------------------------------------------------------------------------------ inputs


@dataclass
class Cloud:
    points: np.ndarray  # (N, 3) float32, display frame (+Y up)
    colors: np.ndarray  # (N, 3) uint8


@dataclass
class Scene:
    name: str
    before: Cloud
    after: Cloud
    bbox_corners: np.ndarray  # (8, 3) corners of bbox_m, display frame
    changes: list[dict]
    to_display: np.ndarray  # (3, 3) recon frame -> display frame (up_axis -> +Y)
    voxel_m: float  # point spacing of the clouds


def point_spacing(points: np.ndarray, n: int = 20_000) -> float:
    """Median nearest-neighbour distance of (a random subset of) the points."""
    if len(points) < 2:
        return 0.01
    sub = points[np.random.default_rng(0).choice(len(points), min(n, len(points)), replace=False)]
    return float(np.median(cKDTree(sub).query(sub, k=2)[0][:, 1]))


def up_rotation(up) -> np.ndarray:
    """Rotation mapping `up` onto +Y (shortest arc, as three.js setFromUnitVectors)."""
    u = np.asarray(up, np.float64)
    u = u / np.linalg.norm(u)
    c = float(u[1])
    if c < -1 + 1e-9:  # half a turn about an axis orthogonal to `up`, as three.js picks it
        axis = np.array([-u[1], u[0], 0.0] if abs(u[0]) > abs(u[2]) else [0.0, -u[2], u[1]])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(u, [0.0, 1.0, 0.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def load_cloud(path: Path, to_display: np.ndarray, max_points: int, seed: int = 0) -> Cloud:
    """Points and colours of a PLY file (grey if it has none), at most `max_points` of them
    (random subset with a fixed seed), rotated into the display frame."""
    vertex = PlyData.read(str(path))["vertex"]
    names = vertex.data.dtype.names
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], 1).astype(np.float32)
    if all(k in names for k in ("red", "green", "blue")):
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], 1)
        if colors.dtype.kind == "f" and colors.size and colors.max() <= 1.0:
            colors = colors * 255
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 170, np.uint8)
    keep = np.flatnonzero(np.isfinite(points).all(1))
    if len(keep) > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(keep, max_points, replace=False))
    return Cloud((points[keep] @ to_display.T.astype(np.float32)), colors[keep])


def box_corners(lo, hi) -> np.ndarray:
    """(8, 3) corners of an axis-aligned box; corner i has bits (x, y, z) = (i>>2, i>>1, i) & 1."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    bits = (np.arange(8)[:, None] >> np.array([2, 1, 0])) & 1
    return np.where(bits == 1, hi, lo)


def load_scene(run: Path, max_points: int) -> Scene:
    viewer = run / "viewer"
    meta = json.loads((viewer / "scene.json").read_text())
    to_display = up_rotation(meta.get("up_axis") or [0.0, -1.0, 0.0])  # viewer's default
    before = load_cloud(viewer / meta.get("before_ply", "before.ply"), to_display, max_points)
    after = load_cloud(viewer / meta.get("after_ply", "after.ply"), to_display, max_points, 1)
    changes_path = run / "changes.json"
    if not changes_path.exists():
        changes_path = viewer / meta.get("changes", "changes.json")
    changes = json.loads(changes_path.read_text()).get("changes") or []
    if meta.get("bbox_m"):
        corners = box_corners(*meta["bbox_m"]) @ to_display.T
    else:  # robust box of both clouds, already in the display frame
        both = np.concatenate([before.points, after.points])
        corners = box_corners(*np.percentile(both, [1, 99], axis=0))
    voxel = (meta.get("stats") or {}).get("voxel_m") or point_spacing(after.points)
    return Scene(meta.get("name") or run.name, before, after, corners, changes, to_display,
                 float(voxel))


# ------------------------------------------------------------------------------ camera


@dataclass
class Camera:
    """Pinhole camera, OpenCV convention (x right, y down, z forward)."""

    R: np.ndarray  # (3, 3) world -> camera, rows = right, down, forward
    eye: np.ndarray  # (3,)
    f: float  # focal length, px
    cx: float
    cy: float
    width: int
    height: int
    flipped: bool = False  # the Z component of VIEW_DIRECTION was flipped

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, np.float64) - self.eye) @ self.R.T

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pixel coordinates u, v (pixel i spans [i, i + 1)) and depth z of world points."""
        cam = self.to_camera(points)
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.f * cam[:, 0] / z + self.cx
            v = self.f * cam[:, 1] / z + self.cy
        return u, v, z


def look_from(direction, corners: np.ndarray, points: np.ndarray, width: int,
              keep: np.ndarray | None = None) -> Camera:
    """Camera at the centre of the box `corners` + d * `direction` (d: the box's bounding
    sphere fills FOV_DEG), focal length and image height fitting the projected `points`
    inside the box (0.5-99.5 percentiles) and all of `keep` with a MARGIN."""
    lo, hi = corners.min(0), corners.max(0)
    centre = 0.5 * (lo + hi)
    direction = np.asarray(direction, np.float64) / np.linalg.norm(direction)
    radius = max(0.5 * float(np.linalg.norm(hi - lo)), 0.1)
    eye = centre + direction * radius / np.sin(np.radians(FOV_DEG) / 2)
    forward = -direction
    right = np.cross(forward, [0.0, 1.0, 0.0])
    if np.linalg.norm(right) < 1e-6:  # looking straight down
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(forward, right), forward])

    def rays(xyz: np.ndarray) -> np.ndarray:  # (n, 2) x / z, y / z in front of the camera
        cam = (np.asarray(xyz, np.float64).reshape(-1, 3) - eye) @ R.T
        cam = cam[cam[:, 2] > NEAR_M]
        return cam[:, :2] / cam[:, 2:]

    inside = rays(points[np.all((points >= lo) & (points <= hi), axis=1)])
    if len(inside) >= 20:  # the points, not the (often empty) corners of the box
        ext = np.percentile(inside, [0.5, 99.5], axis=0)
    else:
        ext = np.stack([rays(corners).min(0), rays(corners).max(0)])
    keep = rays(keep) if keep is not None else np.empty((0, 2))
    if len(keep):
        ext = np.stack([np.minimum(ext[0], keep.min(0)), np.maximum(ext[1], keep.max(0))])
    span_x, span_y = max(ext[1, 0] - ext[0, 0], 1e-6), max(ext[1, 1] - ext[0, 1], 1e-6)
    f = width * (1 - 2 * MARGIN) / span_x
    height = f * span_y / (1 - 2 * MARGIN)
    height = 2 * round(float(np.clip(height, *(np.array(ASPECT_RANGE) * width))) / 2)
    f = min(f, height * (1 - 2 * MARGIN) / span_y)
    cx = width / 2 - f * (ext[0, 0] + ext[1, 0]) / 2
    cy = height / 2 - f * (ext[0, 1] + ext[1, 1]) / 2
    return Camera(R, eye, float(f), float(cx), float(cy), int(width), int(height))


def visible_fraction(cam: Camera, points: np.ndarray, width: int = 200, tol: float = 0.03
                     ) -> float:
    """Fraction of the in-frame points not hidden behind nearer ones: one point per pixel
    of a coarse z-buffer `width` px wide, within `tol` relative depth of the nearest."""
    s = width / cam.width
    w, h = width, max(1, round(cam.height * s))
    u, v, z = cam.project(points)
    with np.errstate(invalid="ignore"):
        ui, vi = np.floor(u * s), np.floor(v * s)
        ok = (z > NEAR_M) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not ok.any():
        return 0.0
    pix = vi[ok].astype(np.int64) * w + ui[ok].astype(np.int64)
    depth = np.full(w * h, np.inf)
    np.minimum.at(depth, pix, z[ok])
    return float(np.mean(z[ok] <= depth[pix] * (1 + tol)))


def choose_camera(corners: np.ndarray, points: np.ndarray, width: int,
                  keep: np.ndarray | None = None, direction=None) -> Camera:
    """View from VIEW_DIRECTION, or from it with the Z component flipped when that view
    sees more of `points` unoccluded (the default would face the back of the filmed
    walls). A given `direction` is used as is."""
    if direction is not None:
        return look_from(direction, corners, points, width, keep)
    flip = np.array(VIEW_DIRECTION) * [1.0, 1.0, -1.0]
    default, other = (look_from(d, corners, points, width, keep) for d in (VIEW_DIRECTION, flip))
    if visible_fraction(other, points) > visible_fraction(default, points):
        other.flipped = True
        return other
    return default


# ----------------------------------------------------------------------------- points


def edl_shade(depth: np.ndarray, strength: float = EDL_STRENGTH) -> np.ndarray:
    """Eye-dome lighting: exp(-strength * mean over the 8 neighbours of max(0, log2 z -
    log2 z_neighbour)); empty pixels (inf) neither get nor cast shade."""
    h, w = depth.shape
    valid = np.isfinite(depth)
    logd = np.zeros(depth.shape, np.float32)
    logd[valid] = np.log2(depth[valid])
    pad_l = np.pad(logd, 1, mode="edge")
    pad_v = np.pad(valid, 1, constant_values=False)
    acc = np.zeros(depth.shape, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            nb = pad_l[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            nv = pad_v[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            acc += np.where(nv & valid, np.maximum(logd - nb, 0.0), 0.0)
    return np.exp(-strength * acc / 8.0)


def fill_holes(depth: np.ndarray, winner: np.ndarray) -> None:
    """In place: an empty pixel with at least 6 of its 8 neighbours filled takes the
    nearest neighbour's point (closes the 1 px gaps between splats)."""
    h, w = depth.shape
    pad_d = np.pad(depth, 1, constant_values=np.inf)
    pad_i = np.pad(winner, 1, constant_values=-1)
    best_d = np.full((h, w), np.inf, np.float32)
    best_i = np.full((h, w), -1, np.int64)
    count = np.zeros((h, w), np.int8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            nd = pad_d[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            ni = pad_i[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            count += np.isfinite(nd)
            closer = nd < best_d
            best_d[closer], best_i[closer] = nd[closer], ni[closer]
    fill = ~np.isfinite(depth) & (count >= 6)
    depth[fill], winner[fill] = best_d[fill], best_i[fill]


def render_cloud(cam: Camera, cloud: Cloud, point_px: int, gamma: float = 1.0,
                 voxel_m: float = 0.0, max_px: int = 6) -> np.ndarray:
    """(H, W, 3) uint8 image of the cloud, z-buffered square splats: each point covers its
    voxel's projected size (`voxel_m` at its depth), at least `point_px`, at most `max_px`."""
    w, h = cam.width, cam.height
    u, v, z = cam.project(cloud.points)
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.clip(np.ceil(1.1 * cam.f * voxel_m / z), point_px, max(point_px, max_px))
        u0 = np.floor(u - size / 2 + 0.5)
        v0 = np.floor(v - size / 2 + 0.5)
        ok = (z > NEAR_M) & (u0 > -size) & (u0 < w) & (v0 > -size) & (v0 < h)
    index = np.flatnonzero(ok)
    u0, v0 = u0[ok].astype(np.int64), v0[ok].astype(np.int64)
    z, size = z[ok].astype(np.float32), size[ok].astype(np.int64)
    depth = np.full(h * w, np.inf, np.float32)
    samples = []  # (pixel, sample -> row of index / z)
    for k in np.unique(size):
        rows = np.flatnonzero(size == k)
        for dv in range(k):
            for du in range(k):
                uu, vv = u0[rows] + du, v0[rows] + dv
                inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                pix, sel = vv[inside] * w + uu[inside], rows[inside]
                np.minimum.at(depth, pix, z[sel])
                samples.append((pix, sel))
    winner = np.full(h * w, -1, np.int64)
    for pix, sel in samples:
        won = z[sel] <= depth[pix]
        winner[pix[won]] = index[sel[won]]
    depth, winner = depth.reshape(h, w), winner.reshape(h, w)
    fill_holes(depth, winner)
    hit = winner.reshape(-1) >= 0
    color = cloud.colors[winner.reshape(-1)[hit]].astype(np.float32) / 255.0
    if gamma != 1.0:
        color **= gamma
    shade = edl_shade(depth).reshape(-1)[hit]
    image = np.empty((h * w, 3), np.float32)
    image[:] = BACKGROUND
    image[hit] = 255.0 * color * shade[:, None]
    return np.clip(image + 0.5, 0, 255).astype(np.uint8).reshape(h, w, 3)


def cloud_gamma(*clouds: Cloud) -> float:
    """Gamma (< 1 brightens) for dark clouds: maps the median luminance towards 115 / 255."""
    lum = np.concatenate([c.colors.astype(np.float32).mean(1) for c in clouds])
    median = float(np.median(lum)) if len(lum) else 255.0
    if median >= DARK_MEDIAN or median <= 1:
        return 1.0
    return float(np.clip(np.log(115 / 255) / np.log(median / 255), 0.7, 1.0))


# ---------------------------------------------------------------------------- overlay


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """DejaVu Sans Bold (system or matplotlib's copy), else PIL's default font."""
    dirs = [Path("/usr/share/fonts/truetype/dejavu")]
    spec = importlib.util.find_spec("matplotlib")
    if spec and spec.origin:
        dirs.append(Path(spec.origin).parent / "mpl-data" / "fonts" / "ttf")
    for folder in dirs:
        for name in FONT_FILES:
            if (folder / name).exists():
                return ImageFont.truetype(str(folder / name), size)
    return ImageFont.load_default(size=size)


def obb_corners(side: dict) -> np.ndarray | None:
    """(8, 3) corners (recon frame) of a change side's oriented box: `obb` (extents
    [w, h, d] rotated about +Y by yaw_deg, three.js convention), else the `aabb`."""
    obb = side.get("obb") if side else None
    if obb:
        half = 0.5 * np.maximum(np.asarray(obb["extents"], np.float64), 0.01)
        yaw = np.radians(obb.get("yaw_deg") or 0.0)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return np.asarray(obb["center"], np.float64) + box_corners(-half, half) @ rot.T
    if side and side.get("aabb"):
        return box_corners(*side["aabb"])
    return None


BOX_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j).bit_count() == 1]


@dataclass
class Tag:
    text: str
    color: tuple[int, int, int]
    top: float  # projected extent of the box (1x px)
    bottom: float
    top_face: tuple[float, float]  # projected centre of the box's top / bottom face
    bottom_face: tuple[float, float]


@dataclass
class Stroke:
    a: tuple[float, float]  # 1x pixel coordinates
    b: tuple[float, float]
    color: tuple[int, int, int]
    width: float  # 1x px
    dashed: bool = False


def clip_project(cam: Camera, a, b) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """2D end points of the 3D segment a-b, clipped to the near plane (None if behind)."""
    ca, cb = cam.to_camera(np.stack([a, b]))
    if ca[2] < NEAR_M and cb[2] < NEAR_M:
        return None
    if ca[2] < NEAR_M:
        ca = ca + (NEAR_M - ca[2]) / (cb[2] - ca[2]) * (cb - ca)
    elif cb[2] < NEAR_M:
        cb = cb + (NEAR_M - cb[2]) / (ca[2] - cb[2]) * (ca - cb)
    pa = (cam.f * ca[0] / ca[2] + cam.cx, cam.f * ca[1] / ca[2] + cam.cy)
    pb = (cam.f * cb[0] / cb[2] + cam.cx, cam.f * cb[1] / cb[2] + cam.cy)
    return pa, pb


def centroid(side: dict, corners: np.ndarray | None) -> np.ndarray | None:
    if side and side.get("centroid"):
        return np.asarray(side["centroid"], np.float64)
    return None if corners is None else corners.mean(0)


class Overlay:
    """Boxes, arrows, tags and the legend on an RGBA layer at SUPERSAMPLE x the size."""

    def __init__(self, cam: Camera, scale: int = SUPERSAMPLE) -> None:
        self.cam, self.s = cam, scale
        self.image = Image.new("RGBA", (cam.width * scale, cam.height * scale), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        unit = cam.width / 1400  # sizes below are for a 1400 px image
        self.line = max(2.0, 3.0 * unit)
        self.font_px = max(10, round(17 * unit))
        self.font = load_font(self.font_px * scale)
        self.small = load_font(max(9, round(15 * unit)) * scale)
        self.taken: list[tuple[float, float, float, float]] = []  # 1x rects of placed tags

    # -- primitives (1x coordinates in, drawn at scale) --
    def _segment(self, a, b, fill, width: float) -> None:
        s = self.s
        a, b = (a[0] * s, a[1] * s), (b[0] * s, b[1] * s)
        w = max(1, round(width * s))
        self.draw.line([a, b], fill=fill, width=w)
        r = w / 2
        for x, y in (a, b):  # round caps: no gaps at box corners
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def _stroke(self, st: Stroke, fill, extra: float = 0.0) -> None:
        if not st.dashed:
            self._segment(st.a, st.b, fill, st.width + extra)
            return
        a, b = np.asarray(st.a), np.asarray(st.b)
        length = float(np.linalg.norm(b - a))
        dash, gap = 3.2 * self.line, 2.2 * self.line
        t = 0.0
        while t < length:
            t1 = min(t + dash, length)
            self._segment(a + (b - a) * t / length, a + (b - a) * t1 / length, fill,
                          st.width + extra)
            t = t1 + gap

    def strokes(self, strokes: list[Stroke], heads: list[tuple[list, tuple]]) -> None:
        """All dark halos first, then the coloured lines (no halo crosses a line)."""
        halo = 2.0 * max(1.0, self.line / 3)
        for st in strokes:
            self._stroke(st, HALO, halo)
        for poly, _ in heads:
            self.draw.polygon([(x * self.s, y * self.s) for x, y in poly], fill=HALO,
                              outline=HALO, width=round(halo * self.s))
        for st in strokes:
            self._stroke(st, (*st.color, 255))
        for poly, color in heads:
            self.draw.polygon([(x * self.s, y * self.s) for x, y in poly], fill=(*color, 255))

    def text_size(self, text: str, font) -> tuple[float, float]:
        x0, y0, x1, y1 = self.draw.textbbox((0, 0), text, font=font, anchor="ls")
        return (x1 - x0) / self.s, (y1 - y0) / self.s

    # -- tags --
    def place_tag(self, text: str, anchor_x: float, top: float, bottom: float
                  ) -> tuple[float, float, float, float]:
        """Rect (1x) of a tag above the box spanning `top`..`bottom` at `anchor_x`, moved
        up past the tags already placed (below the box if it would leave the image)."""
        tw, _ = self.text_size(text, self.font)
        pad = 0.45 * self.font_px
        w, h = tw + 2 * pad, 1.55 * self.font_px
        gap = 0.4 * self.font_px
        x0 = float(np.clip(anchor_x - w / 2, 2, self.cam.width - w - 2))

        def settle(y0: float, step: int) -> float:
            for _ in range(50):
                hits = [r for r in self.taken if x0 < r[2] + 3 and r[0] < x0 + w + 3
                        and y0 < r[3] + 3 and r[1] < y0 + h + 3]
                if not hits:
                    break
                y0 = (min(r[1] for r in hits) - h - 3) if step < 0 else max(r[3] for r in hits) + 3
            return y0

        y0 = settle(top - gap - h, -1)
        if y0 < 2:  # no room above: below the box, moving down
            y0 = settle(bottom + gap, +1)
            if y0 + h > self.cam.height - 2:
                y0 = float(np.clip(top - gap - h, 2, self.cam.height - h - 2))
        rect = (x0, y0, x0 + w, y0 + h)
        self.taken.append(rect)
        return rect

    def draw_tag(self, rect, tag: Tag) -> None:
        """The tag, with a leader line to the box's top (bottom) face centre when it had
        to move away from the box."""
        s, (x0, y0, x1, y1) = self.s, rect
        if y1 <= tag.top + 1:  # above the box
            end, start_y = tag.top_face, y1
        elif y0 >= tag.bottom - 1:  # below it
            end, start_y = tag.bottom_face, y0
        else:
            end, start_y = None, 0.0
        if end is not None and abs(end[1] - start_y) > 1.2 * self.font_px:
            start = (float(np.clip(end[0], x0 + 0.5 * self.font_px, x1 - 0.5 * self.font_px)),
                     start_y)
            self._segment(start, end, (*tag.color, 210), max(1.0, self.line / 2.5))
        self.draw.rounded_rectangle([x0 * s, y0 * s, x1 * s, y1 * s], radius=0.35 * (y1 - y0) * s,
                                    fill=PANEL, outline=(*tag.color, 140), width=max(1, s))
        self.draw.text((0.5 * (x0 + x1) * s, 0.5 * (y0 + y1) * s), tag.text, font=self.font,
                       fill=(*tag.color, 255), anchor="mm")

    # -- legend --
    def legend(self, title: str, counts: dict[str, int], filled: np.ndarray,
               avoid: np.ndarray) -> None:
        """Title and one row per type (box swatch, name, count) in the corner where the
        panel covers the fewest cloud pixels (`filled`) and, much more costly, points of
        `avoid` ((n, 2) samples of the box lines)."""
        s, fp = self.s, self.font_px
        rows = [f"{t}  {counts.get(t, 0)}" for t in TYPES]
        row_h, sw = 1.45 * fp, 1.6 * fp  # row height, swatch width
        pad = 0.75 * fp
        tw = max([self.text_size(title, self.font)[0]]
                 + [sw + 0.6 * fp + self.text_size(r, self.small)[0] for r in rows])
        w, h = tw + 2 * pad, pad * 2 + row_h * (len(rows) + 1) + 0.2 * fp
        m = 0.9 * fp
        origins = [(m if xs == "left" else self.cam.width - w - m,  # ties: first one
                    m if ys == "top" else self.cam.height - h - m)
                   for xs, ys in (("left", "top"), ("left", "bottom"), ("right", "top"),
                                  ("right", "bottom"))]

        def cost(origin: tuple[float, float]) -> float:
            x, y = origin
            cloud = filled[max(int(y), 0) : int(y + h), max(int(x), 0) : int(x + w)]
            hits = np.all((avoid >= (x - m, y - m)) & (avoid <= (x + w + m, y + h + m)), axis=1)
            return round((float(cloud.mean()) if cloud.size else 0.0) + 0.05 * int(hits.sum()), 2)

        x0, y0 = min(origins, key=cost)
        self.draw.rounded_rectangle([x0 * s, y0 * s, (x0 + w) * s, (y0 + h) * s],
                                    radius=0.5 * fp * s, fill=PANEL)
        y = y0 + pad + 0.5 * row_h
        self.draw.text(((x0 + pad) * s, y * s), title, font=self.font, fill=TEXT, anchor="lm")
        y += row_h + 0.2 * fp
        for t, row in zip(TYPES, rows):
            color = TYPE_COLORS[t]
            bx0, bx1 = x0 + pad, x0 + pad + sw
            by0, by1 = y - 0.42 * fp, y + 0.42 * fp
            dashed = t == "unverified"
            for a, b in (((bx0, by0), (bx1, by0)), ((bx1, by0), (bx1, by1)),
                         ((bx1, by1), (bx0, by1)), ((bx0, by1), (bx0, by0))):
                self._stroke(Stroke(a, b, color, 0.75 * self.line, dashed), (*color, 255))
            self.draw.text(((bx1 + 0.6 * fp) * s, y * s), row, font=self.small,
                           fill=(*color, 255) if counts.get(t) else (*color, 150), anchor="lm")
            y += row_h
        self.taken.append((x0, y0, x0 + w, y0 + h))

    def caption(self, name: str, t: float) -> None:
        """Top-left tag "<name>  before  after" with the shown state bright: t = 0 before,
        1 after, in between a cross-fade."""
        s, fp = self.s, self.font_px
        parts = [(name, self.small, 170), ("before", self.font, 70 + 185 * (1 - t)),
                 ("after", self.font, 70 + 185 * t)]
        widths = [self.text_size(text, font)[0] for text, font, _ in parts]
        gap, x0, y0, h = 0.8 * fp, 0.9 * fp, 0.9 * fp, 1.8 * fp
        x1 = x0 + 1.2 * fp + sum(widths) + gap * (len(parts) - 1)
        self.draw.rounded_rectangle([x0 * s, y0 * s, x1 * s, (y0 + h) * s],
                                    radius=0.45 * fp * s, fill=PANEL)
        x = x0 + 0.6 * fp
        for (text, font, alpha), tw in zip(parts, widths):
            self.draw.text((x * s, (y0 + 0.5 * h) * s), text, font=font,
                           fill=(*TEXT, round(alpha)), anchor="lm")
            x += tw + gap

    def composite(self, base: np.ndarray) -> Image.Image:
        layer = self.image.reduce(self.s)  # box filter: line centres keep their exact colour
        return Image.alpha_composite(Image.fromarray(base).convert("RGBA"), layer).convert("RGB")


def change_type(change: dict) -> str:
    return change.get("type") if change.get("type") in TYPES else "unverified"


def change_boxes(scene: Scene) -> list[tuple[dict, str, dict[str, np.ndarray]]]:
    """(change, type, {side: (8, 3) box corners in the display frame}) per change."""
    out = []
    for change in scene.changes:
        sides = {}
        for side in ("before", "after"):
            corners = obb_corners(change.get(side))
            if corners is not None:
                sides[side] = corners @ scene.to_display.T
        out.append((change, change_type(change), sides))
    return out


def draw_overview(base: np.ndarray, cam: Camera, scene: Scene) -> Image.Image:
    """The after render with every change's boxes, arrow and tag, plus the legend."""
    ov = Overlay(cam)
    strokes: list[Stroke] = []  # box edges
    arrows: list[Stroke] = []  # moved: drawn above all boxes
    heads: list[tuple[list, tuple]] = []
    tags: list[Tag] = []
    counts = dict.fromkeys(TYPES, 0)
    for change, kind, sides in change_boxes(scene):
        counts[kind] += 1
        color = TYPE_COLORS[kind]
        if not sides:
            continue
        for side, corners in sides.items():
            secondary = kind == "moved" and side == "before"
            dashed = kind == "unverified" or secondary
            width = ov.line * (0.6 if secondary else 1.0)
            for i, j in BOX_EDGES:
                seg = clip_project(cam, corners[i], corners[j])
                if seg:
                    strokes.append(Stroke(*seg, color, width, dashed))
        if kind == "moved" and len(sides) == 2:
            start = centroid(change.get("before"), sides["before"])
            end = centroid(change.get("after"), sides["after"])
            seg = clip_project(cam, start @ scene.to_display.T, end @ scene.to_display.T)
            if seg:
                (ax, ay), (bx, by) = seg
                length = float(np.hypot(bx - ax, by - ay))
                if length > 1e-3:
                    ux, uy = (bx - ax) / length, (by - ay) / length
                    head = min(3.6 * ov.line, 0.5 * length)
                    base_x, base_y = bx - ux * head, by - uy * head
                    arrows.append(Stroke((ax, ay), (base_x, base_y), color, 1.3 * ov.line))
                    half = 0.55 * head
                    heads.append(([(bx, by), (base_x - uy * half, base_y + ux * half),
                                   (base_x + uy * half, base_y - ux * half)], color))
        main = sides.get("after", sides.get("before"))  # the tag goes to the current place
        _, v, z = cam.project(main)
        if np.any(z <= NEAR_M):
            continue
        faces = []  # projected centres of the two horizontal faces, upper one first
        for idx in ([0, 1, 4, 5], [2, 3, 6, 7]):
            fu, fv, _ = cam.project(main[idx].mean(0, keepdims=True))
            faces.append((float(fu[0]), float(fv[0])))
        faces.sort(key=lambda p: p[1])
        text = f"{change.get('id', '')} {change.get('label') or ''}".strip()[:40]
        tags.append(Tag(text, color, float(v.min()), float(v.max()), faces[0], faces[1]))
    t = np.linspace(0.0, 1.0, 9)[:, None]  # samples along every line, for the legend
    avoid = [np.asarray(st.a) + t * (np.asarray(st.b) - np.asarray(st.a)) for st in strokes]
    avoid += [np.array([[tag.top_face[0], tag.top]]) for tag in tags]
    avoid = np.concatenate(avoid) if avoid else np.empty((0, 2))
    ov.legend(scene.name, counts, np.any(base != BACKGROUND, axis=2), avoid)
    ov.strokes(strokes, [])
    ov.strokes(arrows, heads)
    for tag in sorted(tags, key=lambda t: -t.top):  # lowest box first; later ones move up
        ov.draw_tag(ov.place_tag(tag.text, tag.top_face[0], tag.top, tag.bottom), tag)
    return ov.composite(base)


def draw_state(before: np.ndarray, after: np.ndarray, cam: Camera, name: str, t: float
               ) -> Image.Image:
    """Blend (1 - t) before + t after of the two renders, with the before / after tag."""
    if t <= 0 or t >= 1:
        base = after if t >= 1 else before
    else:
        mix = (1 - t) * before.astype(np.float32) + t * after.astype(np.float32)
        base = np.clip(mix + 0.5, 0, 255).astype(np.uint8)
    ov = Overlay(cam)
    ov.caption(name, t)
    return ov.composite(base)


# ----------------------------------------------------------------------------- output


def quantize(image: Image.Image, colors: int = 256) -> Image.Image:
    """Palette image: median cut, plus exact entries for the overlay colours (a few
    hundred line pixels would otherwise be merged into the cloud's colours)."""
    fixed = [BACKGROUND, TEXT, *TYPE_COLORS.values()]
    adaptive = image.quantize(colors - len(fixed), method=Image.Quantize.MEDIANCUT,
                              dither=Image.Dither.NONE)
    palette = adaptive.getpalette()[: 3 * (colors - len(fixed))] + [v for c in fixed for v in c]
    holder = Image.new("P", (1, 1))
    holder.putpalette(palette)
    return image.quantize(palette=holder, dither=Image.Dither.NONE)


def save_png(image: Image.Image, path: Path, max_bytes: int = MAX_PNG_BYTES) -> None:
    """Optimised PNG; quantised to 256 colours if it is larger than `max_bytes`."""
    image.save(path, optimize=True)
    if path.stat().st_size > max_bytes:
        quantize(image).save(path, optimize=True)


def save_gif(frames: list[Image.Image], path: Path, width: int = GIF_WIDTH,
             max_bytes: int = MAX_GIF_BYTES) -> None:
    """Looping GIF of the cross-fade frames (GIF_DURATIONS_MS), `width` px wide, one palette
    (from the first and the middle frame: before and after) for all frames; fewer colours
    if it would be larger than `max_bytes`."""
    if frames[0].width > width:
        size = (width, round(frames[0].height * width / frames[0].width))
        frames = [f.resize(size, Image.Resampling.LANCZOS) for f in frames]
    ends = Image.new("RGB", (frames[0].width, 2 * frames[0].height))
    ends.paste(frames[0], (0, 0))
    ends.paste(frames[len(frames) // 2], (0, frames[0].height))
    for colors in (256, 128, 64):
        palette = quantize(ends, colors)
        q = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        q[0].save(path, save_all=True, append_images=q[1:], duration=list(GIF_DURATIONS_MS),
                  loop=0, optimize=False, disposal=1)
        if path.stat().st_size <= max_bytes:
            break


def render_run(run: Path, out: Path, width: int = 1400, gif: bool = False,
               max_points: int = 400_000, point_px: int | None = None,
               view: tuple[float, float, float] | None = None) -> dict[str, Path]:
    """Render the images of one run; returns {kind: path}."""
    t0 = time.perf_counter()
    scene = load_scene(run, max_points)
    both = np.concatenate([scene.before.points, scene.after.points])
    if len(both) > 200_000:  # plenty for the framing percentiles and the occlusion test
        both = both[np.random.default_rng(0).choice(len(both), 200_000, replace=False)]
    boxes = [c for _, _, sides in change_boxes(scene) for c in sides.values()]
    keep = np.concatenate(boxes) if boxes else None
    cam = choose_camera(scene.bbox_corners, both, width, keep, view)
    point_px = point_px or (3 if width >= 1000 else 2)
    gamma = cloud_gamma(scene.before, scene.after)
    renders = {r: render_cloud(cam, getattr(scene, r), point_px, gamma, scene.voxel_m)
               for r in ("before", "after")}
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = {k: out.with_name(f"{out.name}_{k}.png") for k in ("overview", "before", "after")}
    save_png(draw_overview(renders["after"], cam, scene), paths["overview"])
    for r, t in (("before", 0.0), ("after", 1.0)):
        save_png(draw_state(renders["before"], renders["after"], cam, scene.name, t), paths[r])
    if gif:  # before, 2 blends, after, 2 blends (then it loops back to before)
        paths["gif"] = out.with_name(f"{out.name}_beforeafter.gif")
        steps = (0.0, 1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3)
        save_gif([draw_state(renders["before"], renders["after"], cam, scene.name, t)
                  for t in steps], paths["gif"])
    print(f"{scene.name}: {cam.width}x{cam.height} px, {point_px} px points, gamma {gamma:.2f}, "
          f"view z {'flipped' if cam.flipped else 'default'}, eye {np.round(cam.eye, 2).tolist()}, "
          f"{len(scene.changes)} changes, {time.perf_counter() - t0:.1f} s")
    for k, p in paths.items():
        print(f"  {k:8s} {p}  {p.stat().st_size / 1e3:.0f} KB")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="run folder, e.g. outputs/scenediff_living_room_1")
    parser.add_argument("--out", type=Path, required=True,
                        help="output prefix, e.g. docs/img/scenediff_living_room_1")
    parser.add_argument("--width", type=int, default=1400, help="image width in px")
    parser.add_argument("--gif", action="store_true", help="also write the cross-fade GIF")
    parser.add_argument("--max-points", type=int, default=400_000, help="per cloud")
    parser.add_argument("--point-px", type=int, default=None,
                        help="splat size in px (default 3 from 1000 px width, else 2)")
    parser.add_argument("--view", type=lambda t: tuple(float(x) for x in t.split(",")),
                        default=None, metavar="X,Y,Z",
                        help="eye direction from the scene centre, +Y up (default: automatic)")
    args = parser.parse_args(argv)
    if args.view is not None and len(args.view) != 3:
        parser.error("--view takes three comma-separated numbers")
    render_run(args.run, args.out, args.width, args.gif, args.max_points, args.point_px,
               args.view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
