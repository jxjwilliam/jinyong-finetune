from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from instruction_jsonl import Pair

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class DedupStats:
    before: int
    after: int

    @property
    def removed(self) -> int:
        return self.before - self.after

    @property
    def removed_ratio(self) -> float:
        if self.before <= 0:
            return 0.0
        return self.removed / float(self.before)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(_normalize_text(text))
    return [t for t in tokens if t.strip()]


def _stable_key(pair: Pair) -> str:
    payload = f"{pair.instruction}\n{pair.input}\n{pair.output}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _to_minhash(tokens: Iterable[str], num_perm: int):
    try:
        from datasketch import MinHash
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "datasketch is required for deduplication. Install with `pip install datasketch`."
        ) from exc
    m = MinHash(num_perm=num_perm)
    for tok in tokens:
        m.update(tok.encode("utf-8"))
    return m


def dedup_continuation_pairs(
    pairs: list[Pair],
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
) -> tuple[list[Pair], DedupStats]:
    try:
        from datasketch import MinHashLSH
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "datasketch is required for deduplication. Install with `pip install datasketch`."
        ) from exc

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[Pair] = []

    for idx, pair in enumerate(pairs):
        joined = f"{pair.input}\n{pair.output}"
        tokens = _tokenize(joined)
        if not tokens:
            kept.append(pair)
            continue

        mh = _to_minhash(tokens, num_perm=num_perm)
        dup_candidates = lsh.query(mh)
        if dup_candidates:
            continue

        key = f"{idx}_{_stable_key(pair)}"
        lsh.insert(key, mh)
        kept.append(pair)

    return kept, DedupStats(before=len(pairs), after=len(kept))

