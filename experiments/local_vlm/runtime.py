"""Offline local-VLM runtime contracts for supported local Qwen VLMs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image, ImageOps

import config


OPTIONS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class OptionPrediction:
    option: str
    confidence: float
    probabilities: dict[str, float]


class OptionRuntime(Protocol):
    model_name: str

    def predict_option(
        self,
        image_path: Path,
        prompt: str,
        *,
        allowed_options: tuple[str, ...] = OPTIONS,
    ) -> OptionPrediction: ...

    def generate_text(self, image_path: Path, prompt: str, *, max_new_tokens: int = 48) -> str: ...


SUPPORTED_QWEN_ARCHITECTURES = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
)


def resize_for_model_input(image: Image.Image) -> Image.Image:
    """Downscale to the shared 1080p budget while preserving aspect ratio."""
    width, height = image.size
    longest = max(width, height)
    pixels = width * height
    scale = min(
        1.0,
        config.LOCAL_VLM_MAX_LONGEST_EDGE / longest,
        (config.LOCAL_VLM_MAX_PIXELS / pixels) ** 0.5,
    )
    if scale >= 1.0:
        return image
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def qwen_model_class_name(config: dict) -> str:
    architectures = config.get("architectures") or []
    supported = [name for name in architectures if name in SUPPORTED_QWEN_ARCHITECTURES]
    if len(supported) != 1:
        raise ValueError(
            f"Expected one of {SUPPORTED_QWEN_ARCHITECTURES}; found {architectures}"
        )
    return supported[0]


def validate_local_qwen_model(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Local Qwen config missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    qwen_model_class_name(config)
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Local Qwen weights index missing under {model_dir}")
    return config


class QwenOptionRuntime:
    """Classify one image-rule prompt using restricted A/B/C/D next-token odds."""

    def __init__(self, model_dir: Path, *, dtype: str = "auto", device_map: str = "auto") -> None:
        model_config = validate_local_qwen_model(model_dir)
        from transformers import (
            AutoProcessor,
            Qwen3_5ForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
        )

        model_classes = {
            "Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGeneration,
            "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
        }
        self.architecture = qwen_model_class_name(model_config)
        model_class = model_classes[self.architecture]

        self.model_dir = model_dir
        self.model_name = model_dir.name
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = model_class.from_pretrained(
            model_dir,
            dtype=dtype,
            device_map=device_map,
            local_files_only=True,
        ).eval()
        option_ids = {
            option: self.processor.tokenizer.encode(option, add_special_tokens=False)
            for option in OPTIONS
        }
        non_single = {option: ids for option, ids in option_ids.items() if len(ids) != 1}
        if non_single:
            raise ValueError(f"A/B/C/D must each be one tokenizer token: {non_single}")
        self.option_token_ids = {option: ids[0] for option, ids in option_ids.items()}

    def _inputs(self, image_path: Path, prompt: str):
        with Image.open(image_path) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            image = resize_for_model_input(oriented)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.architecture == "Qwen3_5ForConditionalGeneration":
            template_kwargs["enable_thinking"] = False
        inputs = self.processor.apply_chat_template(messages, **template_kwargs)
        return inputs.to(self.model.device)

    @torch.inference_mode()
    def predict_option(
        self,
        image_path: Path,
        prompt: str,
        *,
        allowed_options: tuple[str, ...] = OPTIONS,
    ) -> OptionPrediction:
        if not allowed_options or any(option not in OPTIONS for option in allowed_options):
            raise ValueError(f"allowed_options must be a non-empty subset of {OPTIONS}")
        inputs = self._inputs(image_path, prompt)
        outputs = self.model(**inputs, use_cache=False, logits_to_keep=1)
        next_logits = outputs.logits[0, -1]
        option_logits = torch.stack(
            [next_logits[self.option_token_ids[option]] for option in allowed_options]
        )
        probs = torch.softmax(option_logits.float(), dim=0).cpu().tolist()
        probabilities = dict(zip(allowed_options, (float(value) for value in probs), strict=True))
        option = max(probabilities, key=probabilities.__getitem__)
        return OptionPrediction(
            option=option,
            confidence=probabilities[option],
            probabilities=probabilities,
        )

    @torch.inference_mode()
    def generate_text(self, image_path: Path, prompt: str, *, max_new_tokens: int = 48) -> str:
        """Greedy free-form decode (used by the rule retriever's top-K selection,
        which returns several rule numbers in one call instead of one query per rule)."""
        inputs = self._inputs(image_path, prompt)
        generated = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Compatibility alias retained for existing experiment imports.
Qwen35OptionRuntime = QwenOptionRuntime


class GenericOptionRuntime:
    """A/B/C/D next-token odds for any chat VLM exposing the unified HF interface.

    Used for the supplementary edge backbones (Gemma 3n) that are not Qwen but
    load through ``AutoModelForImageTextToText`` + ``AutoProcessor`` with an
    image+text chat template. The decision contract is identical to
    ``QwenOptionRuntime``: one restricted next-token classification over the
    single-token options, plus a greedy free-form decode for the structured arm.
    """

    def __init__(self, model_dir: Path, *, dtype: str = "auto", device_map: str = "auto") -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_dir = model_dir
        self.model_name = model_dir.name
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            dtype=dtype,
            device_map=device_map,
            local_files_only=True,
        ).eval()
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        option_ids = {
            option: tokenizer.encode(option, add_special_tokens=False) for option in OPTIONS
        }
        non_single = {option: ids for option, ids in option_ids.items() if len(ids) != 1}
        if non_single:
            raise ValueError(f"A/B/C/D must each be one tokenizer token: {non_single}")
        self.option_token_ids = {option: ids[0] for option, ids in option_ids.items()}
        self._tokenizer = tokenizer

    def _inputs(self, image_path: Path, prompt: str):
        with Image.open(image_path) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            image = resize_for_model_input(oriented)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    @torch.inference_mode()
    def predict_option(
        self,
        image_path: Path,
        prompt: str,
        *,
        allowed_options: tuple[str, ...] = OPTIONS,
    ) -> OptionPrediction:
        if not allowed_options or any(option not in OPTIONS for option in allowed_options):
            raise ValueError(f"allowed_options must be a non-empty subset of {OPTIONS}")
        inputs = self._inputs(image_path, prompt)
        outputs = self.model(**inputs, use_cache=False)
        next_logits = outputs.logits[0, -1]
        option_logits = torch.stack(
            [next_logits[self.option_token_ids[option]] for option in allowed_options]
        )
        probs = torch.softmax(option_logits.float(), dim=0).cpu().tolist()
        probabilities = dict(zip(allowed_options, (float(value) for value in probs), strict=True))
        option = max(probabilities, key=probabilities.__getitem__)
        return OptionPrediction(
            option=option,
            confidence=probabilities[option],
            probabilities=probabilities,
        )

    @torch.inference_mode()
    def generate_text(self, image_path: Path, prompt: str, *, max_new_tokens: int = 48) -> str:
        inputs = self._inputs(image_path, prompt)
        generated = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class VLLMOptionRuntime:
    """A/B/C/D odds + free-form decode against a remote vLLM OpenAI endpoint.

    Used for a server-class compliance VLM that is served elsewhere (e.g. a
    Qwen3.6 MoE behind vLLM on another host), so it runs in parallel with the
    local GPU. Thinking is disabled so the restricted next-token classification
    sees the option letter directly rather than a chain-of-thought prefix; option
    probability mass is aggregated over the bare and space-prefixed token forms,
    then renormalised over the allowed options to mirror the local runtime.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        model_name: str | None = None,
        enable_thinking: bool = False,
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.model_name = model_name or model_id.split("/")[-1]
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self.max_retries = max_retries

    def _image_data_uri(self, image_path: Path) -> str:
        import base64
        import io

        with Image.open(image_path) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            image = resize_for_model_input(oriented)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _post(self, payload: dict) -> dict:
        import time
        import urllib.error
        import urllib.request

        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * attempt, 8.0))
        raise RuntimeError(f"vLLM request failed after {self.max_retries} attempts: {last_error}")

    def _messages(self, image_path: Path, prompt: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._image_data_uri(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def predict_option(
        self,
        image_path: Path,
        prompt: str,
        *,
        allowed_options: tuple[str, ...] = OPTIONS,
    ) -> OptionPrediction:
        if not allowed_options or any(option not in OPTIONS for option in allowed_options):
            raise ValueError(f"allowed_options must be a non-empty subset of {OPTIONS}")
        result = self._post({
            "model": self.model_id,
            "messages": self._messages(image_path, prompt),
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        })
        content = result["choices"][0].get("logprobs") or {}
        token_entries = content.get("content") or []
        top = token_entries[0]["top_logprobs"] if token_entries else []
        import math

        mass = {option: 0.0 for option in allowed_options}
        for entry in top:
            text = str(entry.get("token", "")).strip().upper()
            if len(text) == 1 and text in mass:
                mass[text] += math.exp(float(entry["logprob"]))
        total = sum(mass.values())
        if total <= 0.0:
            # No option among the returned top tokens; fall back to the decoded letter.
            decoded = str(result["choices"][0]["message"].get("content") or "").strip().upper()
            pick = decoded[:1] if decoded[:1] in mass else allowed_options[0]
            mass = {option: (1.0 if option == pick else 0.0) for option in allowed_options}
            total = 1.0
        probabilities = {option: value / total for option, value in mass.items()}
        option = max(probabilities, key=probabilities.__getitem__)
        return OptionPrediction(
            option=option,
            confidence=probabilities[option],
            probabilities=probabilities,
        )

    def generate_text(self, image_path: Path, prompt: str, *, max_new_tokens: int = 48) -> str:
        result = self._post({
            "model": self.model_id,
            "messages": self._messages(image_path, prompt),
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        })
        return str(result["choices"][0]["message"].get("content") or "").strip()


def build_option_runtime(
    model_dir: Path, *, dtype: str = "auto", device_map: str = "auto"
):
    """Pick the Qwen-specialised or generic runtime from the checkpoint config.

    Qwen VLMs keep their dedicated path (thinking-mode toggle, explicit class);
    every other supported local VLM (Gemma 3n) loads through the generic
    ``AutoModelForImageTextToText`` runtime.
    """
    config_path = model_dir / "config.json"
    architectures = []
    if config_path.is_file():
        architectures = json.loads(config_path.read_text(encoding="utf-8")).get(
            "architectures"
        ) or []
    if any(name in SUPPORTED_QWEN_ARCHITECTURES for name in architectures):
        return QwenOptionRuntime(model_dir, dtype=dtype, device_map=device_map)
    return GenericOptionRuntime(model_dir, dtype=dtype, device_map=device_map)


class DryRunOptionRuntime:
    model_name = "dry-run-local-qwen35"

    def predict_option(
        self,
        image_path: Path,
        prompt: str,
        *,
        allowed_options: tuple[str, ...] = OPTIONS,
    ) -> OptionPrediction:
        del image_path, prompt
        probability = 1.0 / len(allowed_options)
        return OptionPrediction(
            option=allowed_options[-1],
            confidence=probability,
            probabilities={option: probability for option in allowed_options},
        )

    def generate_text(self, image_path: Path, prompt: str, *, max_new_tokens: int = 48) -> str:
        del image_path, prompt, max_new_tokens
        return "1, 2"
