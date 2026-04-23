from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Candidate:
    id: str
    text: str
    sparse_feats: list[float]
    dense_feats: list[float]
    kg_feats: list[float]
    metadata: dict[str, Any] | None = None


@dataclass
class QueryFeatures:
    query_emb: list[float]
    qtype_onehot: list[float]
    kg_coverage: float
    ablation: str | None = None
