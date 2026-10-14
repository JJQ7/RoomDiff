"""Vision-language model wrappers for M5: Qwen3-VL-8B-Instruct (local) and a scripted fake.

Both share `BaseVLM`, which resizes every image to at most `image_max_side` px on the
long side, times each call and appends `{tag, images, prompt, answer, seconds}` to a
JSON-lines log (`open_log`). `ask_json` parses the first JSON object of the reply
(`extract_json`, with a regex fallback for broken JSON) and retries once with
`JSON_RETRY` appended if nothing could be parsed.

`FakeVLM` answers from a script (a callable or a list of strings) so that the semantics
stage can be tested on the CPU without weights.

`quantize_4bit` (config `semantics.quantize_4bit`) loads the language model's linear
layers as bitsandbytes nf4 at load time, everything else (vision tower, merger, embeddings,
head) in bf16: Qwen3-VL-32B-Instruct then takes about 19.5 GiB of GPU memory instead of 65.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
JSON_RETRY = "Answer with JSON only."
_PAIR = re.compile(
    r"""["'](\w+)["']\s*:\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|-?\d+(?:\.\d+)?|true|false|null)""",
    re.IGNORECASE)


# ------------------------------------------------------------------ helpers


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """RGB copy of `image` whose longer side is at most `max_side` px (never upscaled)."""
    image = image.convert("RGB")
    scale = max_side / max(image.size)
    if scale >= 1:
        return image
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _scalar(token: str) -> Any:
    """Value of one regex-matched JSON scalar (quoted string, number, bool, null)."""
    low = token.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "null":
        return None
    if token[0] in "\"'":
        quoted = token if token[0] == '"' else '"' + token[1:-1].replace('"', '\\"') + '"'
        try:
            return json.loads(quoted)
        except json.JSONDecodeError:
            return token[1:-1]
    return float(token) if "." in token else int(token)


def extract_json(text: str) -> dict[str, Any] | None:
    """First JSON object in `text` (code fences and chatter around it are ignored).

    Fallback for broken JSON (single quotes, trailing commas, truncated replies): collect
    every `"key": scalar` pair with a regex. Returns None if nothing usable is found."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    pairs = {key: _scalar(raw) for key, raw in _PAIR.findall(text)}
    return pairs or None


# ------------------------------------------------------------------ base


class BaseVLM:
    """Shared resizing, timing, logging and JSON handling; subclasses implement
    `_generate(images, prompt, max_new_tokens) -> str`."""

    model_id = "base"

    def __init__(self, image_max_side: int = 768, max_new_tokens: int = 256) -> None:
        self.image_max_side = image_max_side
        self.max_new_tokens = max_new_tokens
        self.log_path: Path | None = None
        self.calls = 0
        self.seconds = 0.0

    def open_log(self, path: str | Path) -> None:
        """Start a fresh JSON-lines log of every prompt and answer at `path`."""
        self.log_path = Path(path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("")

    def _generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError

    def ask(self, images: Sequence[Image.Image], prompt: str, tag: str = "",
            names: Sequence[str] = (), max_new_tokens: int | None = None) -> str:
        """Answer `prompt` about `images` (shown as "Image 1:", "Image 2:", ...)."""
        small = [resize_max_side(im, self.image_max_side) for im in images]
        start = time.perf_counter()
        answer = self._generate(small, prompt, max_new_tokens or self.max_new_tokens)
        seconds = time.perf_counter() - start
        self.calls += 1
        self.seconds += seconds
        if self.log_path is not None:
            record = {"tag": tag, "images": list(names), "prompt": prompt, "answer": answer,
                      "seconds": round(seconds, 3)}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return answer

    def ask_json(self, images: Sequence[Image.Image], prompt: str, tag: str = "",
                 names: Sequence[str] = (), max_new_tokens: int | None = None,
                 ) -> dict[str, Any] | None:
        """`ask` and parse the reply as JSON; one retry with `JSON_RETRY` appended."""
        data = extract_json(self.ask(images, prompt, tag, names, max_new_tokens))
        if data is None:
            data = extract_json(self.ask(images, f"{prompt}\n{JSON_RETRY}", f"{tag}:retry",
                                         names, max_new_tokens))
        return data


# ------------------------------------------------------------------ Qwen3-VL


def _drop_page_cache(files: Sequence[Path]) -> None:
    """Ask the kernel to drop the page cache of `files` (WSL2: the guest page cache
    costs host memory, which in turn shrinks the usable VRAM; docs/env_report.md 4)."""
    for file in files:
        try:
            fd = os.open(file, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _weight_files(model_id: str) -> list[Path]:
    """Local safetensors shards of `model_id` (empty if not in the HF cache)."""
    try:
        from huggingface_hub import snapshot_download

        folder = Path(snapshot_download(model_id, local_files_only=True))
    except Exception:  # noqa: BLE001 - best effort only
        return []
    return [p.resolve() for p in folder.glob("*.safetensors")]


def _bnb_config(model_id: str, torch: Any) -> Any:
    """BitsAndBytesConfig: nf4 for the language model's linear layers, everything else
    (vision tower, merger, embeddings, head) left in bf16, listed module by module since
    transformers 5 matches `llm_int8_skip_modules` against full module names."""
    from transformers import AutoConfig, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    with torch.device("meta"):  # no weights, just the module names
        meta = Qwen3VLForConditionalGeneration(AutoConfig.from_pretrained(model_id))
    skip = [name for name, module in meta.named_modules()
            if isinstance(module, torch.nn.Linear) and "language_model" not in name]
    skip += ["model.visual", "visual", "lm_head", "model.language_model.embed_tokens"]
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, llm_int8_skip_modules=skip)


class QwenVLM(BaseVLM):
    """Qwen3-VL (transformers `Qwen3VLForConditionalGeneration`), bf16 on the GPU (or
    its language model in nf4 with `quantize_4bit`), greedy decoding."""

    def __init__(self, model_id: str = QWEN_MODEL, device: str = "cuda",
                 image_max_side: int = 768, max_new_tokens: int = 256,
                 quantize_4bit: bool = False) -> None:
        super().__init__(image_max_side, max_new_tokens)
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.model_id = model_id
        self.quantize_4bit = quantize_4bit
        extra = {"quantization_config": _bnb_config(model_id, torch)} if quantize_4bit else {}
        files = _weight_files(model_id)
        done = threading.Event()

        def dropper() -> None:  # keep the page cache small while the shards stream in
            while not done.wait(1.0):
                _drop_page_cache(files)

        thread = threading.Thread(target=dropper, daemon=True)
        thread.start()
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True,
                **extra)
        finally:
            done.set()
            thread.join()
            _drop_page_cache(files)
        self.model.eval()
        gen = self.model.generation_config  # greedy: clear the sampling defaults
        gen.do_sample, gen.temperature, gen.top_p, gen.top_k = False, None, None, None

    def _generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int) -> str:
        content: list[dict[str, Any]] = []
        for k in range(len(images)):
            content += [{"type": "text", "text": f"Image {k + 1}:"}, {"type": "image"}]
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()


def load_vlm(params: Mapping[str, Any]) -> QwenVLM:
    """Qwen3-VL from the `semantics` config section."""
    return QwenVLM(params["model"], params["device"], params["image_max_side"],
                   params["max_new_tokens"], bool(params.get("quantize_4bit", False)))


# ------------------------------------------------------------------ test fake

FAKE_ANSWER = json.dumps({
    "label": "box", "attributes": "grey cardboard", "description": "A small grey box.",
    "verdict": "confirmed", "reason": "scripted answer", "motion": "rigid",
    "sentences": [], "summary": "A scripted summary of the scene change.",
})

Responder = Callable[[list[Image.Image], str], str]


class FakeVLM(BaseVLM):
    """Scripted VLM for tests: `script` is a callable `(images, prompt) -> answer` or a
    list of answers returned in order (the last one repeats); default `FAKE_ANSWER`,
    which satisfies every JSON schema of the semantics stage. Records every prompt."""

    model_id = "fake"

    def __init__(self, script: Responder | Sequence[str] | None = None,
                 image_max_side: int = 768) -> None:
        super().__init__(image_max_side)
        self.script = script
        self.prompts: list[str] = []
        self.image_sizes: list[list[tuple[int, int]]] = []

    def _generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        self.image_sizes.append([im.size for im in images])
        if self.script is None:
            return FAKE_ANSWER
        if callable(self.script):
            return self.script(images, prompt)
        k = min(len(self.prompts), len(self.script)) - 1
        return self.script[k]
