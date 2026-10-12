"""Model wrappers for M4: SAM 2.1 box/point segmentation and DINOv3 features.

Both are tiny classes with a numpy interface so that the instance stage can be tested on
the CPU with `FakeSegmenter` / `FakeFeaturizer` (no weights, no torch).

* `Segmenter.segment(key, image, box, points)` -> candidate masks (K, H, W) bool and
  their predicted IoU scores (K,). `key` identifies the image so that image embeddings
  can be cached across prompts.
* `Featurizer.patch_tokens(key, image)` -> (gh, gw, D) patch tokens of the whole image
  (resized to multiples of the patch size; the grid covers the image uniformly), and
  `Featurizer.cls_token(image)` -> (D,) global token of a crop.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

SAM_MODEL = "facebook/sam2.1-hiera-large"
DINO_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_PREFIX_TOKENS = 5  # DINOv3 ViT-L/16: 1 CLS + 4 register tokens before the patches


class Segmenter(Protocol):
    def segment(
        self, key: str, image: np.ndarray, box: Sequence[float] | None,
        points: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class Featurizer(Protocol):
    dim: int

    def patch_tokens(self, key: str, image: np.ndarray) -> np.ndarray: ...

    def cls_token(self, image: np.ndarray) -> np.ndarray: ...


def _device(name: str) -> Any:
    import torch

    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[models] CUDA not available, running on the CPU")
        name = "cpu"
    return torch.device(name)


class Sam2Segmenter:
    """SAM 2.1 (transformers `Sam2Model`) prompted with one box and positive points.

    Image embeddings are cached for the last `cache_size` images (about 16 MB each).
    """

    def __init__(self, model_id: str = SAM_MODEL, device: str = "cuda", cache_size: int = 64):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.device = _device(device)
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.cache_size = cache_size
        self.torch = torch

    def _embeddings(self, key: str, image: np.ndarray) -> Any:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        inputs = self.processor(images=image, return_tensors="pt")
        with self.torch.no_grad():
            emb = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device))
        self.cache[key] = emb
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return emb

    def segment(
        self, key: str, image: np.ndarray, box: Sequence[float] | None,
        points: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Three candidate masks (3, H, W) bool and their IoU scores (3,)."""
        height, width = image.shape[:2]
        emb = self._embeddings(key, image)
        kwargs: dict[str, Any] = {}
        if points is not None and len(points):
            kwargs["input_points"] = [[np.asarray(points, float).tolist()]]
            kwargs["input_labels"] = [[[1] * len(points)]]
        if box is not None:
            kwargs["input_boxes"] = [[list(map(float, box))]]
        prompt = self.processor(images=None, original_sizes=[[height, width]],
                                return_tensors="pt", **kwargs)
        tensors = {k: v.to(self.device) for k, v in prompt.items()
                   if k in ("input_points", "input_labels", "input_boxes")}
        with self.torch.no_grad():
            out = self.model(image_embeddings=emb, multimask_output=True, **tensors)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), [[height, width]])[0]
        return masks[0].numpy().astype(bool), out.iou_scores[0, 0].float().cpu().numpy()


class Dinov3Featurizer:
    """DINOv3 ViT (transformers `AutoModel`) patch tokens and CLS tokens, L2-normalised."""

    def __init__(self, model_id: str = DINO_MODEL, device: str = "cuda", cache_size: int = 32):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = _device(device)
        self.model_id = model_id
        proc = AutoImageProcessor.from_pretrained(model_id)
        self.mean = torch.tensor(proc.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(proc.image_std, device=self.device).view(1, 3, 1, 1)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.patch = int(self.model.config.patch_size)
        self.dim = int(self.model.config.hidden_size)
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self.torch = torch

    def _forward(self, image: np.ndarray, size: tuple[int, int]) -> Any:
        """last_hidden_state (1, T, D) of `image` resized to `size` = (H, W)."""
        import torch.nn.functional as F

        x = self.torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
        x = x.permute(2, 0, 1)[None].float() / 255
        x = F.interpolate(x, size=size, mode="bilinear", antialias=True, align_corners=False)
        x = (x - self.mean) / self.std
        with self.torch.no_grad(), self.torch.autocast(self.device.type, dtype=self.torch.bfloat16,
                                                       enabled=self.device.type == "cuda"):
            return self.model(pixel_values=x).last_hidden_state.float()

    def _size(self, height: int, width: int, longest: int | None = None) -> tuple[int, int]:
        if longest is not None:
            scale = longest / max(height, width)
            height, width = height * scale, width * scale
        return (max(1, round(height / self.patch)) * self.patch,
                max(1, round(width / self.patch)) * self.patch)

    def patch_tokens(self, key: str, image: np.ndarray) -> np.ndarray:
        """(gh, gw, D) float32 L2-normalised patch tokens; 378x672 -> 384x672 = 24x42."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        h, w = self._size(*image.shape[:2])
        hidden = self._forward(image, (h, w))[0, DINO_PREFIX_TOKENS:]
        hidden = self.torch.nn.functional.normalize(hidden, dim=-1)
        tokens = hidden.reshape(h // self.patch, w // self.patch, -1).cpu().numpy()
        self.cache[key] = tokens
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return tokens

    def cls_token(self, image: np.ndarray) -> np.ndarray:
        """(D,) L2-normalised CLS token of a crop resized to a longer side of 224 px."""
        hidden = self._forward(image, self._size(*image.shape[:2], longest=224))[0, 0]
        return self.torch.nn.functional.normalize(hidden, dim=0).cpu().numpy()


def load_models(params: Mapping[str, Any]) -> tuple[Sam2Segmenter, Dinov3Featurizer]:
    """SAM 2.1 and DINOv3 from the `instances` config section."""
    return (Sam2Segmenter(params["sam_model"], params["device"]),
            Dinov3Featurizer(params["dino_model"], params["device"]))


# ------------------------------------------------------------------ test fakes


class FakeSegmenter:
    """Returns the prompt box as the mask (or, if `label_maps[key]` is given, the label
    region under most prompt points), with a fixed score. Counts its calls."""

    def __init__(self, label_maps: Mapping[str, np.ndarray] | None = None, score: float = 0.9):
        self.label_maps = dict(label_maps or {})
        self.score = score
        self.calls = 0

    def segment(
        self, key: str, image: np.ndarray, box: Sequence[float] | None,
        points: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        height, width = image.shape[:2]
        mask = np.zeros((height, width), bool)
        labels = self.label_maps.get(key)
        if labels is not None and points is not None and len(points):
            cols = np.clip(np.round(points[:, 0]).astype(int), 0, width - 1)
            rows = np.clip(np.round(points[:, 1]).astype(int), 0, height - 1)
            values, counts = np.unique(labels[rows, cols], return_counts=True)
            mask = labels == values[counts.argmax()]
        elif box is not None:
            x0, y0, x1, y1 = (round(v) for v in box)
            mask[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
        return mask[None], np.array([self.score], np.float32)


class FakeFeaturizer:
    """Colour features: each patch token is the normalised (mean RGB, 0.1) of its patch,
    so equally coloured objects match and differently coloured ones do not."""

    dim = 4

    def __init__(self, patch: int = 16):
        self.patch = patch

    def _feature(self, pixels: np.ndarray) -> np.ndarray:
        feat = np.append(pixels.reshape(-1, 3).mean(0) / 255.0, 0.1)
        return (feat / np.linalg.norm(feat)).astype(np.float32)

    def patch_tokens(self, key: str, image: np.ndarray) -> np.ndarray:
        gh, gw = max(1, image.shape[0] // self.patch), max(1, image.shape[1] // self.patch)
        rows = np.linspace(0, image.shape[0], gh + 1).astype(int)
        cols = np.linspace(0, image.shape[1], gw + 1).astype(int)
        tokens = np.zeros((gh, gw, self.dim), np.float32)
        for r in range(gh):
            for c in range(gw):
                tokens[r, c] = self._feature(image[rows[r]:rows[r + 1], cols[c]:cols[c + 1]])
        return tokens

    def cls_token(self, image: np.ndarray) -> np.ndarray:
        return self._feature(image)
