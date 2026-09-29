"""HuggingFace vision-language adapters: the three internal signals, and where they come from.

This is the only place in the framework that touches a model's internals, so it is
worth being explicit about what each one is and what it costs.

===========================  ==========================================  ==========
signal                       where it comes from                          extra cost
===========================  ==========================================  ==========
``sequence_confidence``      mean probability of the tokens generated     none
``output_entropy``           mean entropy of each step's distribution     none
``visual_attention``         attention mass on the image token span       a forward
===========================  ==========================================  ==========

The first two come out of the same ``generate`` call that produces the text, given
``output_scores=True``. The third needs ``output_attentions=True``, which keeps
every layer's attention matrix alive and is the reason it is opt-in rather than
always on: at 32 layers and a few hundred tokens it is a large amount of memory
for one number.

**A missing reading stays missing.** If a serving stack drops ``output_scores``,
``sequence_confidence`` and ``output_entropy`` come back as ``None`` and the two
signals report themselves unavailable, rather than being handed a placeholder that
would turn them into constant columns. The same applies to attention.

Two architectures are wired up, because the safety family needs a model that
sometimes refuses and not every model does. Nothing in the framework depends on
which one is used: the signals are readings of a response, and the count of them
that run is the same seven either way.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from msrc.types import Sample

__all__ = [
    "HFLLaVAProvider",
    "HFQwenVLProvider",
    "HFVisionLanguageProvider",
    "HFOCRProvider",
]

IMAGE_TOKEN = "<image>"


class HFOCRProvider:
    """RapidOCR behind a tiny interface, or a no-op when it is not installed.

    Kept separate from the VLM because the grounding signal is the one channel
    that needs no GPU at all: it is a separate engine reading pixels, and a
    deployment can run it without the model.
    """

    provider_kind = "rapidocr"

    def __init__(self, engine: Any = None, lang: str = "en"):
        self._engine = engine
        self._lang = lang
        self._cache: Dict[str, List[str]] = {}

    def _load(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def ocr(self, image: str) -> List[str]:
        if not image:
            return []
        if image in self._cache:
            return self._cache[image]
        if not Path(image).is_file():
            self._cache[image] = []
            return []
        engine = self._load()
        result, _ = engine(image)
        texts = [str(item[1]) for item in (result or [])]
        self._cache[image] = texts
        return texts


class HFVisionLanguageProvider:
    """Shared implementation for HuggingFace vision-language checkpoints.

    Subclasses set ``model_class_name`` and ``image_token_fields``; everything else
    -- loading, the generation call, the two distribution readings and the
    attention span -- is architecture-independent, or is written to handle the two
    image-token layouts that the architectures in use here actually produce.

    Parameters
    ----------
    model_path:
        A HuggingFace-format checkpoint directory. Validated before torch is
        imported, so a wrong path reports itself instead of being blamed on a
        missing dependency.
    ocr_provider:
        Supplies the grounding signal. Without one the signal is unavailable.
    want_attention:
        Whether to compute ``visual_attention``. Off by default: it is by far the
        most expensive reading here and the only one that needs a config change.
    """

    provider_kind = "hf-vlm"

    #: Resolved against the ``transformers`` module in :meth:`load`.
    model_class_name = ""

    #: Config attributes that hold the image placeholder's token id, tried in
    #: order. The architectures name it differently and neither name is universal.
    image_token_fields: Sequence[str] = ("image_token_index", "image_token_id")

    def __init__(
        self,
        model_path: str = "",
        ocr_provider: Any = None,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        want_attention: bool = False,
        model_fn: Any = None,
    ):
        self.model_path = model_path
        self.ocr_provider = ocr_provider
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = seed
        self.want_attention = bool(want_attention)
        self._model_fn = model_fn

        self._model = None
        self._processor = None
        self._torch = None
        self.calls = 0
        self.attention_calls = 0
        self._warned_no_attention = False

    # -- loading --------------------------------------------------------
    def image_token_id(self) -> Optional[int]:
        """The placeholder's id, from whichever config field this model uses."""
        config = getattr(self._model, "config", None)
        for field in self.image_token_fields:
            value = getattr(config, field, None)
            if value is not None:
                return int(value)
        return None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path:
            raise ValueError("model_path is required")
        path = Path(self.model_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"model_path is not a directory: {self.model_path}\n"
                "  Pass a HuggingFace-format checkpoint directory, not a file."
            )
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"{self.model_path} holds no config.json, so it is not a HuggingFace "
                "checkpoint directory."
            )
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "the HuggingFace provider needs the model extra:\n"
                '  pip install -e ".[hf]"'
            ) from exc

        model_class = getattr(transformers, self.model_class_name, None)
        if model_class is None:
            raise ImportError(
                f"this transformers build has no {self.model_class_name}. "
                f"Upgrade it, or use a provider whose class it does have."
            )

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": getattr(torch, self.dtype),
            "device_map": self.device,
        }
        if self.want_attention:
            # `output_attentions=True` does nothing under the default `sdpa`
            # attention, and nothing is exactly what it returns: no error, no
            # warning, just an empty attentions tuple. The first full run of this
            # provider asked for attention, counted every pass, and produced a
            # constant column for `visual_attention` on all 800 items because of
            # it. Eager attention is slower, and it is the only implementation that
            # hands the matrices back.
            load_kwargs["attn_implementation"] = "eager"
        self._model = model_class.from_pretrained(self.model_path, **load_kwargs)
        self._model.eval()
        if self.seed is not None:
            # Seeded once, here. Seeding before every generation would make the K
            # resamples identical and the consistency signal would measure nothing.
            torch.manual_seed(int(self.seed))

    # -- generation -----------------------------------------------------
    def generate(self, image: str, prompt: str, do_sample: bool = False) -> Sample:
        if self._model_fn is not None:
            self.calls += 1
            return self._model_fn(image, prompt, do_sample)

        self.load()
        torch = self._torch
        from PIL import Image

        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        text = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        pil = Image.open(image).convert("RGB") if image and Path(image).is_file() else None
        inputs = self._processor(images=pil, text=text, return_tensors="pt").to(self.device)

        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": bool(do_sample),
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        if self.want_attention:
            kwargs["output_attentions"] = True

        with torch.inference_mode():
            output = self._model.generate(**inputs, **kwargs)

        prompt_len = int(inputs["input_ids"].shape[-1])
        gen_ids = output.sequences[0][prompt_len:]
        decoded = self._processor.decode(gen_ids, skip_special_tokens=True).strip()

        confidence, entropy = self._from_scores(output, gen_ids, torch)
        attention = (
            self._visual_attention_mass(
                output, inputs, prompt_len, self.image_token_id()
            )
            if self.want_attention
            else None
        )
        if self.want_attention:
            self.attention_calls += 1
            if attention is None and not self._warned_no_attention:
                # Say it once, and say what to do. A signal that is silently
                # constant is worse than one that is missing: it still occupies a
                # column, and the only place it shows up is a `!!` mark in the
                # evaluation report that nothing forces anyone to read.
                self._warned_no_attention = True
                warnings.warn(
                    "output_attentions=True returned no attention matrices, so "
                    "`visual_attention` will be unavailable for every item. The "
                    "usual cause is an attention implementation that does not "
                    "return them; this provider asks for `eager` when "
                    "want_attention is set, so check that the checkpoint supports "
                    "it. Pass want_attention=False to silence this.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.calls += 1
        return Sample(
            text=decoded,
            sequence_confidence=confidence,
            sequence_entropy=entropy,
            visual_attention_mass=attention,
            meta={"prompt_len": prompt_len, "generated": int(len(gen_ids))},
        )

    @staticmethod
    def _from_scores(output: Any, gen_ids: Any, torch: Any) -> tuple[Optional[float], Optional[float]]:
        """Mean token probability and mean per-step entropy, or ``(None, None)``.

        Returning ``None`` rather than a default is the whole point: a placeholder
        would make both internal signals constant and the calibrator would fit a
        column that encodes nothing.
        """
        scores = getattr(output, "scores", None)
        if not scores:
            return None, None
        probs: List[float] = []
        entropies: List[float] = []
        for step, step_scores in enumerate(scores):
            if step >= len(gen_ids):
                break
            dist = torch.softmax(step_scores[0].float(), dim=-1)
            probs.append(float(dist[int(gen_ids[step])]))
            # Shannon entropy in nats. Clamped away from zero before the log so a
            # one-hot distribution gives exactly 0 rather than a NaN.
            logp = torch.log(dist.clamp_min(1e-12))
            entropies.append(float(-(dist * logp).sum()))
        if not probs:
            return None, None
        return sum(probs) / len(probs), sum(entropies) / len(entropies)

    @staticmethod
    def _visual_attention_mass(
        output: Any, inputs: Any, prompt_len: int, image_token_id: Optional[int]
    ) -> Optional[float]:
        """Fraction of the last layer's attention that lands on image tokens.

        Averages over generated steps in the last layer, then over heads. It is a
        coarse number by construction -- attention is not a saliency map and this
        does not pretend it is -- but "did the answer look at the picture at all"
        is a question a coarse number can answer.

        Two image-token layouts have to be handled, because the architectures in
        use here differ and the difference is invisible from the outside:

        * **Pre-expanded** (Qwen2.5-VL) -- the processor splices one placeholder per
          patch into ``input_ids``, so the image span is exactly the run of
          placeholders and no arithmetic is needed.
        * **Single placeholder** (LLaVA) -- ``input_ids`` holds one ``<image>`` and
          the model expands it internally, so the span starts at the placeholder
          and the number of patches is what the key dimension gained over the
          unexpanded prompt.

        ``image_token_id`` is looked up on the *model's* config by the caller: the
        generation output carries no config, and reading one off it is how this
        returned ``None`` for every item of a full 800-item run while the attention
        matrices were sitting in the output the whole time.
        """
        attentions = getattr(output, "attentions", None)
        if not attentions:
            return None
        if image_token_id is None:
            return None
        ids = inputs["input_ids"][0]
        positions = (ids == int(image_token_id)).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            return None

        first_key_len = int(attentions[0][-1][0].shape[2])
        if len(positions) > 1:
            lo = int(positions.min())
            hi = int(positions.max()) + 1
        else:
            patches = first_key_len - (prompt_len - 1)
            if patches <= 0:
                return None
            lo = int(positions.min())
            hi = lo + patches
        hi = min(hi, first_key_len)
        if hi <= lo:
            return None

        masses: List[float] = []
        for step_attention in attentions:
            layer = step_attention[-1][0]  # last layer, batch 0, [heads, q, k]
            step = layer.shape[1] - 1  # the query that produced the newest token
            row = layer[:, step, :].mean(dim=0)
            span = row[lo:min(hi, row.shape[0])]
            if span.numel() == 0:
                continue
            total = float(row.sum())
            if total <= 0:
                continue
            masses.append(float(span.sum()) / total)
        if not masses:
            return None
        return sum(masses) / len(masses)

    # -- OCR ------------------------------------------------------------
    def ocr(self, image: str) -> List[str]:
        if self.ocr_provider is None:
            return []
        return self.ocr_provider.ocr(image)


class HFLLaVAProvider(HFVisionLanguageProvider):
    """LLaVA-1.5 (or any HF ``LlavaForConditionalGeneration`` checkpoint)."""

    provider_kind = "llava"
    model_class_name = "LlavaForConditionalGeneration"


class HFQwenVLProvider(HFVisionLanguageProvider):
    """Qwen2.5-VL, which refuses some harmful requests where LLaVA refuses none.

    The safety family's label is "did the model refuse", so it needs a model that
    does both. A model that refuses everything and a model that refuses nothing are
    equally unusable there, and this is not a claim about which is better -- it is
    the reason the label distribution has to be checked before a set is collected.

    Defaults to ``bfloat16``, which is the precision this checkpoint was trained in;
    ``float16`` produces degenerate generations on some inputs.
    """

    provider_kind = "qwen-vl"
    model_class_name = "Qwen2_5_VLForConditionalGeneration"

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("dtype", "bfloat16")
        super().__init__(*args, **kwargs)
