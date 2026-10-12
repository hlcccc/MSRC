"""Object-level support: does the model still name an object when it looks again?

The consistency signals in :mod:`msrc.signals` compare whole captions. That is the
wrong resolution for a caption hallucination: a description of a bathroom that
invents a *mirror* and one that invents a *window* agree with each other about the
toilet, the shower and the sink, so caption-level agreement barely moves while the
one claim that is false is exactly the one that matters.

This module drops to the object level instead. It reads the objects the answer
names, then asks of each one: when the same image is looked at again -- four
re-phrasings and three resamples -- how often does this object come back? An
object the model actually saw comes back almost every time; an object it invented
comes back rarely or never. The signal is the **weakest** of those rates, because a
caption is only as trustworthy as its least supported claim.

The lexicon
-----------

Deciding what counts as an object needs a vocabulary, and there is no
domain-neutral one. This ships the official CHAIR synonym list (COCO's 80
categories plus surface forms, one group per line) as
``references/chair_synonyms.txt``. Point ``MSRC_OBJECT_LEXICON`` at a different
file to use another ontology; the format is one group per line, synonyms
comma-separated.

Two things follow, and both belong in any report that quotes this signal:

* On CHAIR the *signal* and the *label* draw on the same ontology -- the label is
  built from COCO's 80 categories, and so is this lexicon. They do not share
  annotations: the label comes from per-image instance annotations, the signal from
  the model's own re-descriptions. Treating the lexicon as a plain noun filter,
  with no synonym merging, still gives AUROC 0.78 rather than 0.89, so part of the
  strength is the ontology and part is the mechanism. Both numbers are honest; the
  split is the reason to report them side by side.
* The signal is *unavailable*, not zero, wherever the answer names no object in the
  lexicon -- a POPE "Yes", a TextVQA reading, a HallusionBench verdict. That is the
  designed behaviour: a constant column would train a calibrator that looks fine
  and means nothing.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_LEXICON_PATH",
    "LEXICON_ENV",
    "ObjectLexicon",
    "clear_cache",
    "default_lexicon",
    "object_min_support",
]

#: Bundled official CHAIR synonym list. Overridable per deployment.
DEFAULT_LEXICON_PATH = (
    Path(__file__).resolve().parent.parent / "references" / "chair_synonyms.txt"
)

#: Environment variable naming an alternative lexicon file.
LEXICON_ENV = "MSRC_OBJECT_LEXICON"

#: A single auxiliary reading gives a degenerate 0/1 support rate, which carries no
#: more information than the caption-level signals already do. Two is the floor at
#: which "the object came back" starts to mean something.
MIN_AUXILIARY = 2


class ObjectLexicon:
    """A set of object groups, each a bag of interchangeable surface forms.

    Groups are the unit of support: "bike" in the caption and "bicycle" in a
    resample are the same object, and counting them as a disagreement would inject
    noise into every reading. The first synonym on a line names the group.
    """

    __slots__ = ("groups", "names", "_word_to_group", "_pattern")

    def __init__(self, groups: Sequence[Sequence[str]]) -> None:
        cleaned: List[Tuple[str, ...]] = []
        for group in groups:
            seen: List[str] = []
            for word in group:
                w = " ".join(str(word).strip().lower().split())
                if w and w not in seen:
                    seen.append(w)
            if seen:
                cleaned.append(tuple(seen))
        if not cleaned:
            raise ValueError("lexicon contains no object groups")

        self.groups: Tuple[Tuple[str, ...], ...] = tuple(cleaned)
        self.names: Tuple[str, ...] = tuple(g[0] for g in self.groups)

        word_to_group: Dict[str, int] = {}
        for i, group in enumerate(self.groups):
            for word in group:
                word_to_group.setdefault(word, i)
        self._word_to_group = word_to_group

        # Longest first so that "wine glass" wins over "glass", and "motor bike"
        # over "bike". Spaces become \s+ so a resample's odd spacing still matches.
        alternatives = sorted(word_to_group, key=len, reverse=True)
        joined = "|".join(
            re.escape(w).replace(r"\ ", r"\s+") for w in alternatives
        )
        self._pattern = re.compile(rf"(?<![a-z0-9])(?:{joined})(?![a-z0-9])")

    # -- construction ---------------------------------------------------

    @classmethod
    def from_path(cls, path: os.PathLike | str) -> "ObjectLexicon":
        """Load a lexicon file: one object group per line, synonyms comma-separated."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"object lexicon not found: {p}")
        groups: List[List[str]] = []
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            groups.append([w for w in line.split(",")])
        if not groups:
            raise ValueError(f"object lexicon is empty: {p}")
        return cls(groups)

    # -- use ------------------------------------------------------------

    def mentioned(self, text: object) -> FrozenSet[int]:
        """Indices of the groups this text names. Case- and spacing-insensitive."""
        haystack = str(text or "").lower()
        if not haystack:
            return frozenset()
        hits = set()
        for match in self._pattern.finditer(haystack):
            group = self._word_to_group.get(" ".join(match.group(0).split()))
            if group is not None:
                hits.add(group)
        return frozenset(hits)

    def name_of(self, index: int) -> str:
        return self.names[index]


@lru_cache(maxsize=4)
def _load(path: str) -> ObjectLexicon:
    return ObjectLexicon.from_path(path)


def default_lexicon() -> ObjectLexicon:
    """The deployment lexicon: ``MSRC_OBJECT_LEXICON``, else the bundled CHAIR list.

    Cached, because the compiled alternation over ~400 surface forms is not
    something to rebuild once per item. The cache is keyed on the resolved path, so
    changing the environment variable takes effect without a restart.
    """
    return _load(os.environ.get(LEXICON_ENV) or str(DEFAULT_LEXICON_PATH))


def clear_cache() -> None:
    """Drop the cached lexicons. For tests and for deployments that swap ontologies."""
    _load.cache_clear()


def object_min_support(
    primary_text: object,
    auxiliary_texts: Sequence[object],
    lexicon: ObjectLexicon,
    *,
    min_readings: int = MIN_AUXILIARY,
) -> Optional[Tuple[float, int, str]]:
    """Lowest cross-reading mention rate among the objects ``primary_text`` names.

    Returns ``(min_support, n_named_objects, weakest_object)``, or ``None`` when the
    reading cannot be taken -- no object named, or too few auxiliary readings. The
    caller turns ``None`` into *unavailable* rather than into a risk of zero.

    ``min_readings`` exists because the floor is a property of *where the readings
    come from*, not of the arithmetic. Two readings of the same model is the floor at
    which repeating yourself starts to mean something; **one** reading from a
    *different* model is already independent evidence, and refusing it would throw
    away the only cross-model object-level reading a two-model deployment has.
    ``msrc.signals.CrossModelObjectSupport`` is the caller that lowers it.
    """
    if min_readings < 1:
        raise ValueError("min_readings must be at least 1")
    named = lexicon.mentioned(primary_text)
    if not named:
        return None
    if len(auxiliary_texts) < min_readings:
        return None

    aux_sets = [lexicon.mentioned(t) for t in auxiliary_texts]
    n_aux = len(aux_sets)
    worst_rate = 1.0
    worst_index = -1
    for group in sorted(named):
        hits = sum(1 for s in aux_sets if group in s)
        rate = hits / n_aux
        if rate < worst_rate:
            worst_rate, worst_index = rate, group
    return worst_rate, len(named), lexicon.name_of(worst_index)
