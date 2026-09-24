"""HuggingFace dataset loaders.

Adapts real corpora from the HuggingFace Hub to the :class:`BaseDataset`
interface: one AI-generated corpus plus one human control corpus, so metrics
stop being toy-only. (The non-native-English corpus for false-positive-bias
measurement is a separate Phase-2 issue.)

Requires the ``models`` extra (``pip install -e '.[models]'``) for the
``datasets`` library. The Hub import is deferred to first iteration and the
materialized examples are cached on the instance, so iterating stays
repeatable and never re-downloads.
"""

from __future__ import annotations

from typing import Iterator, Sequence

from .._deps import require
from .base import BaseDataset, Example


class HFDataset(BaseDataset):
    """A single HuggingFace Hub corpus mapped onto :class:`Example` records.

    Every row gets the same ``label`` -- pair an AI corpus (``label=1``) with
    a human corpus (``label=0``) via :func:`load_hf_dataset` for a mixed
    benchmark set.

    Args:
        hf_name: Dataset repo on the Hub, e.g. ``"ag_news"``.
        label: Fixed label for every example: ``1`` = AI, ``0`` = human.
        split: Hub split to load, e.g. ``"train"``.
        limit: Maximum examples to keep (first rows); ``None`` keeps all.
        text_field: Column holding the document text. Rows with a missing or
            blank value in this column are skipped.
        source: Provenance tag on each example; defaults to ``hf_name``.
    """

    name = "hf"

    def __init__(
        self,
        hf_name: str,
        label: int = 0,
        split: str = "train",
        limit: int | None = None,
        text_field: str = "text",
        source: str | None = None,
    ) -> None:
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {label!r}")
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit!r}")
        self.hf_name = hf_name
        self.label = label
        self.split = split
        self.limit = limit
        self.text_field = text_field
        self.source = source if source is not None else hf_name
        self._examples: list[Example] | None = None

    def _load(self) -> list[Example]:
        """Download the split and materialize examples, once."""
        if self._examples is not None:
            return self._examples

        datasets = require("datasets")
        ds = datasets.load_dataset(self.hf_name, split=self.split)
        if self.text_field not in ds.column_names:
            raise ValueError(
                f"column {self.text_field!r} not in {self.hf_name!r} "
                f"(available: {', '.join(ds.column_names)})"
            )

        examples: list[Example] = []
        for row in ds:
            text = row.get(self.text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            examples.append(
                Example(text=text, label=self.label, source=self.source)
            )
            if self.limit is not None and len(examples) >= self.limit:
                break
        self._examples = examples
        return examples

    def __iter__(self) -> Iterator[Example]:
        return iter(self._load())


class _CombinedDataset(BaseDataset):
    """Concatenate several datasets, preserving order.

    Iteration is repeatable because each wrapped dataset is repeatable.
    """

    name = "hf-mixed"

    def __init__(self, datasets: Sequence[BaseDataset]) -> None:
        self._datasets = list(datasets)

    def __iter__(self) -> Iterator[Example]:
        for dataset in self._datasets:
            yield from dataset


def load_hf_dataset(
    ai_dataset: str = "tatsu-lab/alpaca",
    human_dataset: str = "ag_news",
    *,
    split: str = "train",
    limit: int | None = None,
    ai_text_field: str = "output",
    human_text_field: str = "text",
) -> BaseDataset:
    """Load one AI-generated corpus and one human corpus from the HF Hub.

    Returns a :class:`BaseDataset` yielding AI examples (``label=1``) from
    ``ai_dataset`` followed by human examples (``label=0``) from
    ``human_dataset``, each tagged with its corpus as ``source``. Iteration is
    repeatable; pass ``limit`` to cap examples *per corpus* and keep runs fast.

    The defaults pair GPT-3-generated instructions (``tatsu-lab/alpaca``)
    with human-written news (``ag_news``) -- both public, no auth required.

    Args:
        ai_dataset: Hub repo for the AI-generated corpus.
        human_dataset: Hub repo for the human-written corpus.
        split: Hub split to load from both corpora.
        limit: Maximum examples per corpus; ``None`` keeps everything.
        ai_text_field: Text column in the AI corpus.
        human_text_field: Text column in the human corpus.
    """
    return _CombinedDataset(
        [
            HFDataset(
                ai_dataset,
                label=1,
                split=split,
                limit=limit,
                text_field=ai_text_field,
            ),
            HFDataset(
                human_dataset,
                label=0,
                split=split,
                limit=limit,
                text_field=human_text_field,
            ),
        ]
    )
