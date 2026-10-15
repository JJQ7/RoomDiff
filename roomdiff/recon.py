"""M2 recon: Pi3X reconstruction of both keyframe sets in one shared metric frame.

`joint` (default): the keyframes of both recordings (all before, then all after) go
through ONE Pi3X forward pass. Pi3X is permutation-equivariant over its input views, so
both camera sets come out in one shared frame without a registration step.

`separate` (fallback): one pass per recording; `after` is then moved into the `before`
frame by a Sim(3) found through a third, small joint pass on `n_anchor` keyframes of each
recording (Umeyama on the anchor cameras' centres and look-at points), optionally refined
by point-to-plane ICP.

Levelling (`level_world`, default on): the model's frame is arbitrary (roughly the first
camera), so the result is rotated into a gravity-aligned frame: +Y up (camera up estimate
refined by a RANSAC floor plane), the floor at y = 0 and the first before-camera looking
along +Z (projected onto the floor), see `level_transform`. With `snap_to_walls` (off by
default) the frame is then turned about +Y so that the dominant wall direction lies along
+X / +Z (`wall_yaw`), which makes object boxes room-aligned.

Writes `<out>/recon/recon.npz`, `meta.json`, `before.ply`, `after.ply`
(docs/interfaces.md, M2) and prints a one-line alignment verdict (`alignment_verdict`:
OK, FAIR = a warning, WEAK = the two reconstructions do not line up; M3 then demotes its
removed / added candidates to unverified only with `visibility.weak_alignment_demotes`,
off by default).
"""

from __future__ import annotations

import functools
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree

from roomdiff.geometry import (
    estimate_up,
    project_points,
    rotation_between,
    rotation_y,
    se3_inverse,
    sim3_apply_to_poses,
    sim3_matrix,
    transform_points,
    umeyama,
    unproject_depth,
    voxel_downsample,
)
from roomdiff.paths import RECORDINGS, OutputPaths

if TYPE_CHECKING:
    import torch

PATCH = 14  # Pi3X patch size: image sides must be multiples of it
ICP_MAX_DIST = 0.05  # m, max ICP correspondence distance
# alignment verdict, on co-visible points (see `alignment_verdict`), both directions:
VERDICT_OK, VERDICT_FAIR, VERDICT_WEAK = "alignment OK", "alignment FAIR", "alignment WEAK"
OK_MEDIAN = 0.02  # m, max median nearest-neighbour distance
OK_FRACTION = 0.9  # min fraction of points within 10 cm of the other cloud
MIN_COVISIBLE = 0.1  # min fraction of a recording's points the other one could observe
# FAIR, a warning only (nothing downstream changes): the two rules above pass, but in one
# direction fewer than this fraction of the co-visible points lie within 3 cm of the other
# cloud. Measured (weaker direction): good runs 85-98 %; the tabletop runs with one
# recording turned a quarter turn 74 % (the whole table top came out "added", 7 and 4 false
# positives) although their median (1.0-1.6 cm) and 10 cm share (95-97 %) passed; but
# three SceneDiff pairs without such errors measure 77-79 % (a fourth 80 %), so FAIR only
# flags and only WEAK acts (visibility).
FAIR_FRACTION_3CM = 0.8
# levelling (`level_transform`)
LEVEL_STRIDE = 4  # floor fit on every 4th valid pixel (rows and columns)
FLOOR_BAND = 10.0  # floor candidates: the lowest 10 % of the points along the camera up
FLOOR_MAX_TILT_DEG = 20.0  # a floor normal farther than this from the camera up is ignored
FLOOR_INLIER_M = 0.02  # m, RANSAC inlier distance to the floor plane
RANSAC_ITERATIONS = 500
RANSAC_MAX_POINTS = 50_000  # floor candidates scored by RANSAC (random subset)
# wall snapping (`wall_yaw`), in the levelled frame
WALL_BAND_M = (0.3, 2.0)  # wall points: this high above the floor (no floor, few tabletops)
WALL_VOXEL_M = 0.05  # m, the band is voxel-downsampled before the normals are estimated
WALL_NEIGHBOURS = 16  # local PCA over this many nearest voxel centres
WALL_PLANARITY = 0.1  # planar: smallest PCA eigenvalue below this times the middle one
WALL_MAX_TILT_DEG = 15.0  # a wall normal lies within this angle of the horizontal plane
WALL_PEAK_HALF_WIDTH = 5  # deg, histogram peak = the 1-degree bins within +-5 of the best one
WALL_MIN_VOTES = 50  # fewer wall-like points: no snapping

# recon.npz contract (docs/interfaces.md): key -> dtype
RECON_DTYPES: dict[str, Any] = {
    "rec": np.uint8,
    "files": np.str_,
    "images": np.uint8,
    "cam2world": np.float32,
    "K": np.float32,
    "depth": np.float32,
    "conf": np.float16,
    "valid": np.bool_,
}


# ----------------------------------------------------------------- preprocessing


def target_size(width: int, height: int, pixel_limit: int = 255000) -> tuple[int, int]:
    """Model input size `(W, H)`: the rule of `pi3.utils.basic.load_images_as_tensor`.

    Scale to an area of about `pixel_limit`, round both sides to multiples of 14, then
    shrink the side that deviates most from the aspect ratio until the area fits.
    """
    scale = math.sqrt(pixel_limit / (width * height))
    w, h = width * scale, height * scale
    k, m = round(w / PATCH), round(h / PATCH)
    while (k * PATCH) * (m * PATCH) > pixel_limit:
        if k / m > w / h:
            k -= 1
        else:
            m -= 1
    return max(1, k) * PATCH, max(1, m) * PATCH


def preprocess_images(
    paths: Sequence[str | Path], pixel_limit: int = 255000
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Load keyframes and resize them all to one model size (from the first image).

    Returns `(tensor (N,3,H,W) float in [0,1], images (N,H,W,3) uint8, scales (N,2))`
    where `scales[i] = (W / width_i, H / height_i)`. Resizing uses PIL Lanczos exactly
    like Pi3's own loader. Raises ValueError if the aspect ratios differ.
    """
    import torch
    from PIL import Image

    images, scales = [], []
    size: tuple[int, int] | None = None
    first = (0, 0, Path())
    for path in paths:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if size is None:
                size, first = target_size(w, h, pixel_limit), (w, h, Path(path))
            elif abs(w * first[1] - h * first[0]) > 0.01 * first[0] * h:  # > 1 %
                raise ValueError(
                    f"keyframes have different aspect ratios: {first[2].name} is "
                    f"{first[0]}x{first[1]} but {Path(path).name} is {w}x{h}. One Pi3X pass "
                    "needs one image size: record both videos in the same orientation "
                    "(both portrait or both landscape) and the same camera mode."
                )
            images.append(np.asarray(im.resize(size, Image.Resampling.LANCZOS)))
            scales.append((size[0] / w, size[1] / h))
    if not images:
        raise ValueError("no keyframes to reconstruct")
    array = np.stack(images)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().float().div(255)
    return tensor, array, np.asarray(scales, np.float32)


# ---------------------------------------------------------------------- model


def load_model(
    name: str = "yyfz233/Pi3X", device: str = "cuda", head_chunk: int = 16
) -> torch.nn.Module:
    """Load Pi3X from the Hugging Face cache directly onto the GPU.

    The model is built on the GPU and the safetensors weights are streamed there, so the
    5.4 GB fp32 checkpoint is never held twice in CPU RAM (as `from_pretrained` would).
    The unused multimodal branches (depth / ray / pose priors) are dropped afterwards;
    without priors they contribute exactly zero. `head_chunk`: see `set_head_chunk`.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from pi3.models.pi3x import Pi3X
    from safetensors.torch import load_model as load_safetensors

    if not torch.cuda.is_available():
        raise RuntimeError("Pi3X reconstruction needs a CUDA GPU")
    weights = hf_hub_download(name, "model.safetensors")
    with torch.device(device):
        model = Pi3X(use_multimodal=True)
    load_safetensors(model, weights, strict=True, device=device)
    model.eval()
    model.disable_multimodal()
    set_head_chunk(model, head_chunk)
    return model


def set_head_chunk(model: torch.nn.Module, chunk: int) -> None:
    """Run Pi3X's point and confidence heads on `chunk` frames at a time.

    `Pi3X.forward_head` calls `self._chunked_conv_head(head, feat, patch_h, patch_w)`
    and so always gets its default `chunk_size=64`. These full-resolution conv heads set
    the peak VRAM: 94 frames at 378x672 peak at 19.5 GB allocated with 64 and 12.4 GB
    with 16, at the same speed. Shadowing the method on this instance with a smaller
    `chunk_size` does this without editing third_party; results do not change (median
    alignment distances within 0.005 cm) since every frame goes through the heads alone.
    """
    if chunk < 1:
        raise ValueError(f"recon.head_chunk must be positive, got {chunk}")
    method = type(model)._chunked_conv_head  # type: ignore[attr-defined]
    model._chunked_conv_head = functools.partial(method, model, chunk_size=chunk)


def run_pi3x(
    images: torch.Tensor,
    model: torch.nn.Module,
    conf_threshold: float = 0.25,
    edge_rtol: float = 0.03,
) -> dict[str, Any]:
    """One Pi3X forward pass over all `images` (N,3,H,W) in [0,1].

    Returns numpy arrays in the model's metric frame: `cam2world` (N,4,4), `K` (N,3,3)
    (recovered from the predicted rays, pixel units), `depth` (N,H,W) z-depth, `conf`
    (N,H,W) sigmoid probability, `valid` (N,H,W) = conf > threshold and not a
    depth/normal edge, and the float `metric_scale` the model applied.
    """
    import torch
    from pi3.utils.geometry import depth_normal_edge, recover_intrinsic_from_rays_d

    device = next(model.parameters()).device
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            res = model(images[None].to(device))
        local = res["local_points"][0].float()
        conf = torch.sigmoid(res["conf"][0, ..., 0].float())
        K = recover_intrinsic_from_rays_d(res["rays"][0].float())
        cam2world = res["camera_poses"][0].float()
        metric = float(res["metric"][0])
        del res
        confident = conf > conf_threshold
        valid = confident & ~depth_normal_edge(local, rtol=edge_rtol, mask=confident)
        out = {
            "cam2world": cam2world.cpu().numpy(),
            "K": K.cpu().numpy(),
            "depth": local[..., 2].cpu().numpy(),
            "conf": conf.cpu().numpy(),
            "valid": valid.cpu().numpy(),
            "metric_scale": metric,
        }
    del local, conf, confident, valid
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ separate mode


def anchor_indices(n: int, k: int) -> np.ndarray:
    """Up to `k` evenly spaced indices in `range(n)`, first and last included."""
    return np.unique(np.linspace(0, n - 1, min(k, n)).round().astype(int))


def anchor_points(pred: Mapping[str, Any], frames: Iterable[int]) -> np.ndarray:
    """Camera centre and look-at point (centre + optical axis * median valid depth) of
    each frame, stacked (2F, 3). The look-at points constrain the rotation even when the
    camera centres are nearly collinear."""
    points = []
    for i in frames:
        T, depth, valid = pred["cam2world"][i], pred["depth"][i], pred["valid"][i]
        d = float(np.median(depth[valid] if valid.any() else depth))
        points += [T[:3, 3], T[:3, 3] + d * T[:3, 2]]
    return np.asarray(points, np.float64)


def icp_refine(
    source: np.ndarray, target: np.ndarray, max_dist: float = ICP_MAX_DIST
) -> tuple[np.ndarray, dict[str, float]]:
    """Point-to-plane ICP (Open3D, Tukey loss) of `source` onto `target` (both (M,3)).

    Returns the rigid 4x4 correction and fitness / inlier RMSE before and after.
    """
    import open3d as o3d

    reg = o3d.pipelines.registration

    def cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        return pc

    src, dst = cloud(source), cloud(target)
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist, max_nn=30))
    initial = reg.evaluate_registration(src, dst, max_dist, np.eye(4))
    result = reg.registration_icp(
        src, dst, max_dist, np.eye(4),
        reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=max_dist / 2)),
        reg.ICPConvergenceCriteria(max_iteration=50),
    )
    T = np.asarray(result.transformation)
    angle = math.degrees(math.acos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1.0, 1.0)))
    info = {
        "fitness_before": initial.fitness,
        "rmse_before_m": initial.inlier_rmse,
        "fitness_after": result.fitness,
        "rmse_after_m": result.inlier_rmse,
        "translation_m": float(np.linalg.norm(T[:3, 3])),
        "rotation_deg": angle,
    }
    return T, info


def align_separate(
    images: torch.Tensor,
    rec: np.ndarray,
    model: torch.nn.Module,
    n_anchor: int = 8,
    icp: bool = True,
    voxel: float = 0.02,
    **pi3x_kwargs: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate passes per recording, expressed in the `before` frame.

    Returns the concatenated prediction (like `run_pi3x`, before frames first) and a
    dict describing the alignment (Sim(3), anchor residuals, ICP result).
    """
    import torch

    frames = [np.flatnonzero(rec == r) for r in range(len(RECORDINGS))]
    preds = [run_pi3x(images[torch.from_numpy(f)], model, **pi3x_kwargs) for f in frames]
    local = [anchor_indices(len(f), n_anchor) for f in frames]
    joint_idx = np.concatenate([f[a] for f, a in zip(frames, local)])
    joint = run_pi3x(images[torch.from_numpy(joint_idx)], model, **pi3x_kwargs)

    to_joint, info = [], {"method": "separate", "anchors": [len(a) for a in local]}
    offset = 0
    for name, pred, a in zip(RECORDINGS, preds, local):
        src = anchor_points(pred, a)
        dst = anchor_points(joint, range(offset, offset + len(a)))
        offset += len(a)
        s, R, t = umeyama(src, dst)
        residual = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        info[f"anchor_rmse_{name}_m"] = float(np.sqrt((residual**2).mean()))
        to_joint.append(sim3_matrix(s, R, t))
    M = np.linalg.inv(to_joint[0]) @ to_joint[1]  # after frame -> before frame
    after = preds[1]
    after["cam2world"], scale = sim3_apply_to_poses(M, after["cam2world"])
    after["depth"] = after["depth"] * scale
    info["sim3_after_to_before"] = {"scale": scale, "matrix": M.round(6).tolist()}

    if icp:
        clouds = [voxel_downsample(valid_points(p, range(len(f)))[0], None, voxel)[0]
                  for p, f in zip(preds, frames)]
        T, icp_info = icp_refine(clouds[1], clouds[0])
        icp_info["accepted"] = icp_info["fitness_after"] >= icp_info["fitness_before"]
        if icp_info["accepted"]:
            after["cam2world"], _ = sim3_apply_to_poses(T, after["cam2world"])
        info["icp"] = icp_info

    pred = {k: np.concatenate([p[k] for p in preds]) for k in preds[0] if k != "metric_scale"}
    pred["metric_scale"] = preds[0]["metric_scale"]
    info["metric_scale_per_pass"] = [p["metric_scale"] for p in (*preds, joint)]
    return pred, info


# ------------------------------------------------------------------- levelling


def fit_floor(
    points: np.ndarray, up: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, float, float] | None:
    """Floor plane among the lowest `FLOOR_BAND` % of `points` (M, 3) along `up`.

    MSAC (RANSAC scored by the truncated squared distance, cap `FLOOR_INLIER_M`) over
    planes whose normal lies within `FLOOR_MAX_TILT_DEG` of `up`; points more than the cap
    *below* a plane cost twice the cap, since the floor is the lowest large plane (this
    stops a slightly tilted plane through part of the floor and some wall bottoms from
    winning). Then least squares on all points within `FLOOR_INLIER_M` of the plane, kept
    if it scores no worse. Returns `(normal (towards up), offset, inlier fraction of the
    band)` with the floor at `normal @ x = offset`, or None if no such plane exists.
    """
    heights = points @ up
    band = points[heights <= np.percentile(heights, FLOOR_BAND)].astype(np.float64)
    if len(band) < 3:
        return None
    rng = np.random.default_rng(seed)
    sample = band[rng.choice(len(band), RANSAC_MAX_POINTS, replace=False)] \
        if len(band) > RANSAC_MAX_POINTS else band
    min_cos = math.cos(math.radians(FLOOR_MAX_TILT_DEG))

    def cost(normal: np.ndarray, offset: float) -> float:
        r = sample @ normal - offset
        cap = FLOOR_INLIER_M**2
        return float(np.minimum(r**2, cap).sum() + cap * (r < -FLOOR_INLIER_M).sum())

    best, best_cost = None, math.inf
    for _ in range(RANSAC_ITERATIONS):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm if normal @ up >= 0 else -normal / norm
        if normal @ up < min_cos:
            continue
        hypothesis = (normal, float(normal @ a))
        if (value := cost(*hypothesis)) < best_cost:
            best, best_cost = hypothesis, value
    if best is None:
        return None
    normal, offset = best
    near = points[np.abs(points @ normal - offset) < FLOOR_INLIER_M].astype(np.float64)
    centre = near.mean(0)
    refined = np.linalg.svd(near - centre, full_matrices=False)[2][-1]
    refined = refined if refined @ up > 0 else -refined
    if refined @ up >= min_cos and cost(refined, float(refined @ centre)) <= best_cost:
        normal, offset = refined, float(refined @ centre)
    inliers = float(np.mean(np.abs(band @ normal - offset) < FLOOR_INLIER_M))
    return normal, offset, inliers


def wall_yaw(points: np.ndarray) -> tuple[float, float, int] | None:
    """Dominant horizontal wall direction of a levelled cloud (M, 3) (+Y up, floor y = 0).

    Points `WALL_BAND_M` above the floor are voxel-downsampled (`WALL_VOXEL_M`); each gets
    a normal by local PCA over its `WALL_NEIGHBOURS` nearest neighbours. Planar
    neighbourhoods (`WALL_PLANARITY`) whose normal is within `WALL_MAX_TILT_DEG` of the
    horizontal vote with the normal's yaw folded modulo 90 degrees (opposite and
    perpendicular walls agree) into a circular histogram of 1-degree bins. The peak is the
    window of +-`WALL_PEAK_HALF_WIDTH` bins with the most votes; its yaw is the circular
    mean (of 4 x angle) of the votes in it. Returns `(yaw_deg in (-45, 45], peak fraction =
    votes in the window / all votes, votes)`, or None without votes. Yaw convention of
    `rotation_y` / `floor_obb`: `rotation_y(-yaw_deg)` turns the walls onto the axes.
    """
    band = points[(points[:, 1] >= WALL_BAND_M[0]) & (points[:, 1] <= WALL_BAND_M[1])]
    pts = voxel_downsample(band, None, WALL_VOXEL_M)[0].astype(np.float64)
    if len(pts) <= WALL_NEIGHBOURS:
        return None
    _, idx = cKDTree(pts).query(pts, WALL_NEIGHBOURS)
    local = pts[idx] - pts[idx].mean(1, keepdims=True)
    eigval, eigvec = np.linalg.eigh(np.einsum("nki,nkj->nij", local, local))
    normal = eigvec[:, :, 0]  # eigenvector of the smallest eigenvalue
    wall = (eigval[:, 0] < WALL_PLANARITY * eigval[:, 1]) & (
        np.abs(normal[:, 1]) < math.sin(math.radians(WALL_MAX_TILT_DEG)))
    if not wall.any():
        return None
    angles = np.degrees(np.arctan2(-normal[wall, 2], normal[wall, 0])) % 90.0
    hist = np.bincount(np.floor(angles).astype(int) % 90, minlength=90)
    window = np.arange(-WALL_PEAK_HALF_WIDTH, WALL_PEAK_HALF_WIDTH + 1)
    best = int(np.argmax([hist[(b + window) % 90].sum() for b in range(90)]))
    offset = (angles - (best + 0.5) + 45.0) % 90.0 - 45.0  # signed, folded distance to the peak
    inside = np.abs(offset) <= WALL_PEAK_HALF_WIDTH + 0.5
    mean = np.exp(1j * np.radians(4 * angles[inside])).mean()
    yaw = -((-math.degrees(np.angle(mean)) / 4 + 45.0) % 90.0 - 45.0)  # in (-45, 45]
    return yaw + 0.0, float(inside.mean()), int(wall.sum())


def level_transform(
    recon: Mapping[str, Any], snap_to_walls: bool = False, wall_peak_fraction: float = 0.25
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rigid 4x4 transform from the model frame into the levelled world frame.

    Up: the camera estimate (`estimate_up`), replaced by the normal of the RANSAC floor
    plane (`fit_floor`) when there is one. The transform maps up to +Y, the floor plane
    to y = 0 (without a plane: the median height of the lowest 5 % of the points) and turns
    about Y so that the first before-camera's optical axis projects onto +Z. With
    `snap_to_walls` it then turns about Y by -`wall_yaw` (dominant wall direction onto
    +X / +Z) if that peak holds at least `wall_peak_fraction` of the votes. Uses
    `cam2world`, `depth`, `K`, `valid`, `rec`. Returns the transform and a description.
    """
    cam2world = np.asarray(recon["cam2world"], np.float64)
    camera_up = estimate_up(cam2world)
    s = LEVEL_STRIDE
    points = np.concatenate([np.zeros((0, 3), np.float32)] + [
        unproject_depth(recon["depth"][i], recon["K"][i], cam2world[i])[::s, ::s][
            recon["valid"][i][::s, ::s]] for i in range(len(cam2world))]).astype(np.float64)
    floor = fit_floor(points, camera_up) if len(points) else None
    if floor is None:
        up, inliers, method = camera_up, None, "camera up (no floor plane)"
        heights = points @ up if len(points) else np.zeros(1)
        offset = float(np.median(heights[heights <= np.percentile(heights, 5.0)]))
    else:
        (up, offset, inliers), method = floor, "ransac floor plane"
    R = rotation_between(up, [0.0, 1.0, 0.0])
    first = cam2world[int(np.argmax(np.asarray(recon["rec"]) == 0))]  # first before-camera
    forward = R @ first[:3, 2]
    yaw = math.degrees(math.atan2(forward[0], forward[2])) if np.hypot(
        forward[0], forward[2]) > 1e-6 else 0.0
    T = np.eye(4)
    T[:3, :3] = rotation_y(-yaw) @ R
    T[1, 3] = -offset
    tilt = math.degrees(math.acos(float(np.clip(camera_up @ up, -1.0, 1.0))))
    info: dict[str, Any] = {
        "up_axis_original": np.round(up, 5).tolist(),
        "floor_inliers_fraction": None if inliers is None else round(inliers, 4),
        "method": method,
        "camera_up": np.round(camera_up, 5).tolist(),
        "floor_vs_camera_up_deg": round(tilt, 2),
        "floor_offset_m": round(offset, 4),
        "points_used": len(points),
    }
    if snap_to_walls:
        walls = wall_yaw(transform_points(T, points)) if len(points) else None
        snapped = walls is not None and walls[2] >= WALL_MIN_VOTES and \
            walls[1] >= wall_peak_fraction
        if snapped:
            turn = np.eye(4)
            turn[:3, :3] = rotation_y(-walls[0])
            T = turn @ T
        info.update({"walls_snapped": snapped,
                     "wall_yaw_deg": None if walls is None else round(walls[0], 2),
                     "wall_peak_fraction": None if walls is None else round(walls[1], 4),
                     "wall_votes": 0 if walls is None else walls[2]})
    info["transform"] = T.round(6).tolist()
    return T, info


# ----------------------------------------------------------------- point clouds


def valid_points(
    recon: Mapping[str, Any], frames: Iterable[int]
) -> tuple[np.ndarray, np.ndarray | None]:
    """World points (M,3) of the valid pixels of `frames`, plus their RGB colours (M,3)
    uint8 if `recon` has `images`."""
    points, colors = [np.zeros((0, 3), np.float32)], [np.zeros((0, 3), np.uint8)]
    for i in frames:
        mask = recon["valid"][i]
        world = unproject_depth(recon["depth"][i], recon["K"][i], recon["cam2world"][i])
        points.append(world[mask])
        if "images" in recon:
            colors.append(recon["images"][i][mask])
    return np.concatenate(points), np.concatenate(colors) if "images" in recon else None


def recording_points(recon: Mapping[str, Any], rec_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Valid world points and colours of one recording (0 = before, 1 = after)."""
    points, colors = valid_points(recon, np.flatnonzero(recon["rec"] == rec_id))
    assert colors is not None
    return points, colors


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Binary little-endian PLY with float xyz and uint8 rgb."""
    vertex = np.empty(len(points), [("xyz", "<f4", 3), ("rgb", "u1", 3)])
    vertex["xyz"], vertex["rgb"] = points, colors
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertex.tofile(f)


# ----------------------------------------------------------------- diagnostics


def _nn_stats(distances: np.ndarray) -> dict[str, float]:
    return {
        "median_m": float(np.median(distances)),
        "mean_m": float(np.mean(distances)),
        "frac_lt_3cm": float(np.mean(distances < 0.03)),
        "frac_lt_10cm": float(np.mean(distances < 0.10)),
    }


def observable_by(
    points: np.ndarray,
    recon: Mapping[str, Any],
    rec_id: int,
    rel_tol: float = 0.05,
    abs_tol: float = 0.05,
) -> np.ndarray:
    """Mask of the world `points` (M,3) that recording `rec_id` could have observed.

    A point is observable if it projects into a valid pixel of some keyframe of that
    recording and is not behind the surface seen there (depth <= seen depth * (1 +
    rel_tol) + abs_tol). Points in front of a seen surface stay observable: that space
    was seen empty, so a point there is a disagreement (a change or a misalignment).
    """
    h, w = recon["depth"].shape[1:]
    points = np.asarray(points, np.float64)
    seen = np.zeros(len(points), bool)
    for i in np.flatnonzero(recon["rec"] == rec_id):
        world2cam = se3_inverse(recon["cam2world"][i].astype(np.float64))
        u, v, z = project_points(points, recon["K"][i].astype(np.float64), world2cam)
        with np.errstate(invalid="ignore"):
            col, row = np.round(u), np.round(v)
            inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        idx = np.flatnonzero(inside & ~seen)
        row, col = row[idx].astype(int), col[idx].astype(int)
        depth = recon["depth"][i][row, col]
        ok = recon["valid"][i][row, col] & (z[idx] <= depth * (1 + rel_tol) + abs_tol)
        seen[idx[ok]] = True
    return seen


def alignment_diagnostics(
    recon: Mapping[str, Any], voxel: float = 0.02
) -> tuple[dict[str, Any], list[tuple[np.ndarray, np.ndarray]]]:
    """Cross-recording consistency of a reconstruction.

    Voxel-downsamples the valid points of each recording and measures nearest-neighbour
    distances before->after and after->before (median, mean, fraction < 3 / 10 cm):
    `nn_all` over all points, `nn_covisible` over the points the other recording could
    have observed (`observable_by`; `fraction` = share of points kept). Areas filmed by
    one video only dominate `nn_all`; the verdict (`alignment_verdict`: OK / FAIR / WEAK,
    with `verdict_reason` and `frac_lt_3cm_min`) uses `nn_covisible`, where only real
    changes and misalignment remain. Also reports the scene extent (1st-99th percentile
    box), camera path length, mean confidence and valid fraction per recording.
    Returns the diagnostics and the two downsampled coloured clouds.
    """
    clouds = [voxel_downsample(*recording_points(recon, r), voxel)
              for r in range(len(RECORDINGS))]
    pts = [c[0] for c in clouds]
    diag: dict[str, Any] = {"voxel_m": voxel, "nn_all": {}, "nn_covisible": {}}
    if all(len(p) for p in pts):
        for a, b in ((0, 1), (1, 0)):
            key = f"{RECORDINGS[a]}_to_{RECORDINGS[b]}"
            dist = cKDTree(pts[b]).query(pts[a], workers=8)[0]
            diag["nn_all"][key] = _nn_stats(dist)
            mask = observable_by(pts[a], recon, b)
            if mask.any():
                diag["nn_covisible"][key] = {"fraction": float(mask.mean()),
                                             **_nn_stats(dist[mask])}
        lo, hi = np.percentile(np.concatenate(pts), [1, 99], axis=0)
        diag["bbox_p1_p99_m"] = [lo.round(3).tolist(), hi.round(3).tolist()]
        diag["scene_extent_m"] = (hi - lo).round(3).tolist()
    for r, name in enumerate(RECORDINGS):
        sel = recon["rec"] == r
        centres = recon["cam2world"][sel, :3, 3]
        diag[name] = {
            "frames": int(sel.sum()),
            "camera_path_m": float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum()),
            "mean_conf": float(recon["conf"][sel].astype(np.float32).mean()),
            "valid_fraction": float(recon["valid"][sel].mean()),
            "points_voxelized": len(pts[r]),
        }
    verdict, reason, frac_min = alignment_verdict(diag["nn_covisible"])
    diag.update(verdict=verdict, verdict_reason=reason, frac_lt_3cm_min=frac_min)
    return diag, clouds


def frac_lt_3cm_min(nn_covisible: Mapping[str, Mapping[str, float]]) -> float | None:
    """Smaller of the two directions' fractions of co-visible points within 3 cm of the
    other cloud (`nn_covisible` of `alignment_diagnostics`); None without any."""
    fractions = [d["frac_lt_3cm"] for d in nn_covisible.values() if "frac_lt_3cm" in d]
    return min(fractions) if fractions else None


def alignment_verdict(
    nn_covisible: Mapping[str, Mapping[str, float]],
) -> tuple[str, str, float | None]:
    """Verdict, reason and `frac_lt_3cm_min` from the co-visible statistics of both
    directions (`nn_covisible` of `alignment_diagnostics`, also as stored in
    recon/meta.json).

    WEAK: a direction has no co-visible points or covers less than MIN_COVISIBLE of its
    recording, or its median exceeds OK_MEDIAN or less than OK_FRACTION of it lies within
    10 cm (M3 then demotes its blobs to unverified, `visibility.weak_alignment_demotes`).
    FAIR (a warning): not WEAK, but `frac_lt_3cm_min` < FAIR_FRACTION_3CM. Else OK.
    """
    covis = [nn_covisible.get(k) for k in ("before_to_after", "after_to_before")]
    frac_min = frac_lt_3cm_min(nn_covisible)
    if not all(c and c["fraction"] >= MIN_COVISIBLE for c in covis):
        return VERDICT_WEAK, "little co-visible overlap", frac_min
    if not all(c["median_m"] <= OK_MEDIAN and c["frac_lt_10cm"] >= OK_FRACTION
               for c in covis if c):
        return VERDICT_WEAK, (f"co-visible median > {100 * OK_MEDIAN:.0f} cm or < "
                              f"{100 * OK_FRACTION:.0f}% within 10 cm"), frac_min
    if frac_min is not None and frac_min < FAIR_FRACTION_3CM:
        return VERDICT_FAIR, (f"< {100 * FAIR_FRACTION_3CM:.0f}% of the co-visible points "
                              "within 3 cm in one direction"), frac_min
    return VERDICT_OK, "", frac_min


def verdict_line(diag: Mapping[str, Any]) -> str:
    """One-line summary of `alignment_diagnostics`."""
    parts = []
    for key, d in diag["nn_covisible"].items():
        parts.append(
            f"{key.replace('_to_', '->')}: {100 * d['fraction']:.0f}% co-visible, median "
            f"{100 * d['median_m']:.1f} cm, <3cm {100 * d['frac_lt_3cm']:.0f}%, "
            f"<10cm {100 * d['frac_lt_10cm']:.0f}%"
        )
    if diag.get("frac_lt_3cm_min") is not None:
        parts.append(f"min <3cm {100 * diag['frac_lt_3cm_min']:.0f}%")
    reason = f"{diag['verdict_reason']}; " if diag["verdict_reason"] else ""
    return f"{diag['verdict']} ({reason}{'; '.join(parts) or 'no co-visible points'})"


# ----------------------------------------------------------------------- I/O


def write_recon(path: str | Path, recon: Mapping[str, Any]) -> None:
    """Write `recon.npz` (uncompressed) with the dtypes of the interface contract."""
    n, h, w = np.shape(recon["depth"])
    expected = {"rec": (n,), "files": (n,), "images": (n, h, w, 3), "cam2world": (n, 4, 4),
                "K": (n, 3, 3), "depth": (n, h, w), "conf": (n, h, w), "valid": (n, h, w)}
    arrays = {}
    for key, dtype in RECON_DTYPES.items():
        arrays[key] = np.asarray(recon[key]).astype(dtype, copy=False)
        if arrays[key].shape != expected[key]:
            raise ValueError(f"recon[{key!r}] has shape {arrays[key].shape}, "
                             f"expected {expected[key]}")
    np.savez(path, metric_scale=np.float64(recon["metric_scale"]), **arrays)


def load_recon(path: str | Path) -> dict[str, Any]:
    """Read `recon.npz`; `metric_scale` is returned as a Python float."""
    with np.load(path) as f:
        recon = {key: f[key] for key in f.files}
    recon["metric_scale"] = float(recon["metric_scale"])
    return recon


# ---------------------------------------------------------------------- stage


def _cuda_peak_gb(reset: bool = False) -> dict[str, float] | None:
    """Peak VRAM since the last reset, or None if CUDA was never initialised."""
    import torch

    if not torch.cuda.is_initialized():
        return None
    if reset:
        torch.cuda.reset_peak_memory_stats()
        return None
    return {"allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


def run_recon(out_dir: str | Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the keyframes in `<out>/keyframes/` and write `<out>/recon/`.

    `cfg` is the full config; only its `recon` section is used. Returns `meta.json`.
    """
    params = cfg["recon"]
    paths = OutputPaths(out_dir)
    index = json.loads(paths.keyframes_index.read_text())
    files = [paths.keyframes_of(r) / f["file"] for r in RECORDINGS for f in index[r]["frames"]]
    counts = [len(index[r]["frames"]) for r in RECORDINGS]
    if len(files) > params["max_frames"]:
        raise ValueError(f"{len(files)} keyframes exceed recon.max_frames="
                         f"{params['max_frames']}; lower ingest.num_keyframes")
    rec = np.repeat(np.arange(len(RECORDINGS), dtype=np.uint8), counts)
    times: dict[str, float] = {}
    start = time.perf_counter()

    tensor, images, _ = preprocess_images(files, params["pixel_limit"])
    times["preprocess"] = time.perf_counter() - start
    t0 = time.perf_counter()
    model = load_model(params["model"], head_chunk=params["head_chunk"])
    times["load_model"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _cuda_peak_gb(reset=True)
    kwargs = {"conf_threshold": params["conf_threshold"], "edge_rtol": params["edge_rtol"]}
    if params["mode"] == "joint":
        pred = run_pi3x(tensor, model, **kwargs)
        alignment: dict[str, Any] = {"method": "joint"}
    elif params["mode"] == "separate":
        pred, alignment = align_separate(
            tensor, rec, model, params["n_anchor"], params["icp_refine"],
            params["voxel_size_diag"], **kwargs,
        )
    else:
        raise ValueError(f"recon.mode must be 'joint' or 'separate', got {params['mode']!r}")
    times["inference"] = time.perf_counter() - t0
    peak = _cuda_peak_gb()
    del model, tensor

    level: dict[str, Any] = {"levelled": False}
    if params["level_world"]:
        t0 = time.perf_counter()
        T, info = level_transform({**pred, "rec": rec}, params["snap_to_walls"],
                                  params["wall_peak_fraction"])
        pred["cam2world"], _ = sim3_apply_to_poses(T, pred["cam2world"])
        level = {"levelled": True, "up_axis_original": info.pop("up_axis_original"),
                 "floor_inliers_fraction": info.pop("floor_inliers_fraction"),
                 **{k: info.pop(k) for k in ("walls_snapped", "wall_yaw_deg",
                                             "wall_peak_fraction") if k in info},
                 "level": info}
        times["level"] = time.perf_counter() - t0
    scale = params["scale_factor"]
    pred["cam2world"][:, :3, 3] *= scale
    pred["depth"] = pred["depth"] * scale
    recon = {"rec": rec, "files": np.array([p.name for p in files]), "images": images, **pred}
    paths.recon.mkdir(parents=True, exist_ok=True)
    write_recon(paths.recon / "recon.npz", recon)

    t0 = time.perf_counter()
    diag, clouds = alignment_diagnostics(recon, params["voxel_size_diag"])
    for name, (points, colors) in zip(RECORDINGS, clouds):
        write_ply(paths.recon / f"{name}.ply", points, colors)
    times["diagnostics"] = time.perf_counter() - t0
    times["total"] = time.perf_counter() - start

    n, h, w = pred["depth"].shape
    meta = {
        "model": params["model"],
        "mode": params["mode"],
        "resolution": [w, h],
        "frames": dict(zip(RECORDINGS, counts)),
        "metric_scale": pred["metric_scale"],
        "scale_factor": scale,
        "params": dict(params),
        "runtime_s": {k: round(v, 2) for k, v in times.items()},
        "peak_vram_gb": peak,
        "alignment": alignment,
        **level,
        "diagnostics": diag,
    }
    (paths.recon / "meta.json").write_text(json.dumps(meta, indent=2))
    vram = f", peak VRAM {peak['allocated']:.1f} GB" if peak else ""
    print(f"[recon] {params['mode']}: {n} frames at {w}x{h}, metric scale "
          f"{pred['metric_scale']:.3f}, inference {times['inference']:.1f} s{vram}, "
          f"total {times['total']:.1f} s -> {paths.recon}")
    print(f"[recon] {verdict_line(diag)}")
    if level["levelled"]:
        info, fraction = level["level"], level["floor_inliers_fraction"]
        walls = ""
        if level.get("wall_yaw_deg") is not None:
            walls = (f"; walls at {level['wall_yaw_deg']:+.1f} deg "
                     f"({100 * level['wall_peak_fraction']:.0f}% of {info['wall_votes']} votes), "
                     + ("snapped to the axes" if level["walls_snapped"] else "not snapped"))
        print(f"[recon] levelled: {info['method']}, "
              f"{info['floor_vs_camera_up_deg']:.1f} deg from the camera up"
              + (f", {100 * fraction:.0f}% floor inliers" if fraction is not None else "")
              + walls)
    return meta


def is_levelled(recon_dir: str | Path) -> bool:
    """`recon/meta.json` says the world frame is levelled (+Y up, floor at y = 0)."""
    meta = Path(recon_dir) / "meta.json"
    return meta.is_file() and bool(json.loads(meta.read_text()).get("levelled", False))
