"""Back-translation attack via Helsinki-NLP MarianMT.

Round-trips text ``EN -> pivot language -> EN`` so the paraphrase preserves
meaning while changing surface form -- the classic "laundering" attack on
AI-text detectors. The selling point is fluency: unlike character noise, the
output should read like ordinary (if slightly bland) prose.

Requires the ``models`` extra (``pip install -e '.[models]'``). The heavy
imports are deferred to first use and both translation models are cached on
the instance, per the model-backed component conventions in CONTRIBUTING.md.
"""

from __future__ import annotations

import re

from .._deps import require
from .base import BaseAttack


# Split on sentence-ending punctuation followed by whitespace. opus-mt models
# are trained on single sentences: feeding a whole paragraph lets greedy
# decoding emit EOS early and silently drop trailing sentences, so the attack
# round-trips sentence by sentence and reassembles in order.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences on ``.``/``!``/``?`` boundaries."""
    return [s for s in _SENTENCE_END_RE.split(text.strip()) if s]


class BackTranslationAttack(BaseAttack):
    """Paraphrase text with an EN -> pivot -> EN MarianMT round-trip.

    Args:
        pivot: Pivot language code, e.g. ``"fr"``. Helsinki-NLP must publish
            ``opus-mt-en-{pivot}`` and ``opus-mt-{pivot}-en`` models.
        max_length: Maximum token length for translation inputs and outputs.
            Longer inputs are truncated.
    """

    name = "back_translation"

    def __init__(self, pivot: str = "fr", max_length: int = 256) -> None:
        self.pivot = pivot
        self.max_length = max_length
        self._torch = None
        #: (tokenizer, model) for EN -> pivot; loaded lazily by :meth:`_load`.
        self._forward = None
        #: (tokenizer, model) for pivot -> EN; loaded lazily by :meth:`_load`.
        self._backward = None

    def _load(self) -> None:
        """Import torch/transformers and load both MarianMT models, once."""
        if self._forward is not None:
            return

        torch = require("torch")
        transformers = require("transformers")

        self._torch = torch
        self._forward = (
            transformers.MarianTokenizer.from_pretrained(
                f"Helsinki-NLP/opus-mt-en-{self.pivot}"
            ),
            transformers.MarianMTModel.from_pretrained(
                f"Helsinki-NLP/opus-mt-en-{self.pivot}"
            ),
        )
        self._backward = (
            transformers.MarianTokenizer.from_pretrained(
                f"Helsinki-NLP/opus-mt-{self.pivot}-en"
            ),
            transformers.MarianMTModel.from_pretrained(
                f"Helsinki-NLP/opus-mt-{self.pivot}-en"
            ),
        )
        self._forward[1].eval()
        self._backward[1].eval()

    def _translate(self, text: str, tokenizer, model) -> str:
        """Translate ``text`` with one MarianMT model (greedy = deterministic)."""
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        with self._torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=1,  # greedy decoding: deterministic for a fixed model
                do_sample=False,
            )
        return tokenizer.decode(generated[0], skip_special_tokens=True)

    def _round_trip(self, text: str) -> str:
        """Translate one sentence ``text`` EN -> pivot -> EN."""
        forward_tokenizer, forward_model = self._forward
        backward_tokenizer, backward_model = self._backward
        pivoted = self._translate(text, forward_tokenizer, forward_model)
        return self._translate(pivoted, backward_tokenizer, backward_model)

    def transform(self, text: str) -> str:
        """Return the EN -> pivot -> EN paraphrase of ``text``.

        The text is split into sentences first: opus-mt is a sentence-level
        model, and greedy-decoding a whole paragraph can emit EOS early and
        drop trailing sentences. Each sentence is round-tripped independently
        and the results are reassembled in the original order.
        """
        if not text.strip():
            return text
        self._load()
        return " ".join(self._round_trip(s) for s in _split_sentences(text))
