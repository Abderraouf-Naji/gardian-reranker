from __future__ import annotations

from typing import Any

import torch

from .model import GardianModel
from .schemas import Candidate, QueryFeatures


class GardianReranker:
    """Drop-in reranker API for external RAG pipelines."""

    def __init__(self, model: GardianModel, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: dict[str, Any], device: str = "cpu") -> "GardianReranker":
        model = GardianModel(
            sparse_dim=int(config.get("sparse_dim", 3)),
            dense_dim=int(config.get("dense_dim", 4)),
            kg_dim=int(config.get("kg_dim", 6)),
            branch_hidden=int(config.get("branch_hidden", 128)),
            controller_hidden=int(config.get("controller_hidden", 128)),
            query_feat_dim=int(config.get("query_feat_dim", 768)),
            n_qtypes=int(config.get("n_qtypes", 7)),
            dropout=float(config.get("dropout", 0.1)),
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        return cls(model=model, device=device)

    def rerank(self, query_features: QueryFeatures, candidates: list[Candidate]) -> dict[str, Any]:
        if not candidates:
            return {"query_weights": {"sparse": 0.0, "dense": 0.0, "kg": 0.0}, "ranked": []}

        to_t = lambda arr: torch.tensor(arr, dtype=torch.float32, device=self.device)
        sparse = to_t([c.sparse_feats for c in candidates])
        dense = to_t([c.dense_feats for c in candidates])
        kg = to_t([c.kg_feats for c in candidates])
        q_emb = to_t([query_features.query_emb] * len(candidates))
        qtype = to_t([query_features.qtype_onehot] * len(candidates))
        kg_cov = to_t([query_features.kg_coverage] * len(candidates))

        with torch.no_grad():
            scores, weights, breakdown = self.model(
                sparse_feats=sparse,
                dense_feats=dense,
                kg_feats=kg,
                query_emb=q_emb,
                qtype_onehot=qtype,
                kg_coverage=kg_cov,
                ablation=query_features.ablation,
                return_breakdown=True,
            )

        score_v = scores.cpu().tolist()
        weight_v = weights[0].cpu().tolist()
        sparse_c = breakdown["sparse_contrib"].cpu().tolist()
        dense_c = breakdown["dense_contrib"].cpu().tolist()
        kg_c = breakdown["kg_contrib"].cpu().tolist()

        ranked = []
        for i, cand in enumerate(candidates):
            ranked.append(
                {
                    "id": cand.id,
                    "text": cand.text,
                    "score": float(score_v[i]),
                    "contrib": {
                        "sparse": float(sparse_c[i]),
                        "dense": float(dense_c[i]),
                        "kg": float(kg_c[i]),
                    },
                    "metadata": cand.metadata or {},
                }
            )
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return {
            "query_weights": {
                "sparse": float(weight_v[0]),
                "dense": float(weight_v[1]),
                "kg": float(weight_v[2]),
            },
            "ranked": ranked,
        }
