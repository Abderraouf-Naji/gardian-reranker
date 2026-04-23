"""Standalone GARDIAN model for reusable reranking."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ControllerMLP(nn.Module):
    def __init__(self, query_feat_dim: int, n_qtypes: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        in_dim = query_feat_dim + n_qtypes + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, query_emb: torch.Tensor, qtype_onehot: torch.Tensor, kg_coverage: torch.Tensor) -> torch.Tensor:
        x = torch.cat([query_emb, qtype_onehot, kg_coverage.unsqueeze(-1)], dim=-1)
        return F.softmax(self.net(x), dim=-1)


class GardianModel(nn.Module):
    def __init__(
        self,
        sparse_dim: int = 3,
        dense_dim: int = 4,
        kg_dim: int = 6,
        branch_hidden: int = 128,
        controller_hidden: int = 128,
        query_feat_dim: int = 768,
        n_qtypes: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sparse_head = BranchMLP(sparse_dim, branch_hidden, dropout)
        self.dense_head = BranchMLP(dense_dim, branch_hidden, dropout)
        self.kg_head = BranchMLP(kg_dim, branch_hidden, dropout)
        self.controller = ControllerMLP(query_feat_dim, n_qtypes, controller_hidden, dropout)

    def forward(
        self,
        sparse_feats: torch.Tensor,
        dense_feats: torch.Tensor,
        kg_feats: torch.Tensor,
        query_emb: torch.Tensor,
        qtype_onehot: torch.Tensor,
        kg_coverage: torch.Tensor,
        ablation: Optional[str] = None,
        return_breakdown: bool = False,
    ):
        s_sparse = self.sparse_head(sparse_feats)
        s_dense = self.dense_head(dense_feats)
        s_kg = self.kg_head(kg_feats)

        qtype_in = torch.zeros_like(qtype_onehot) if ablation == "no_qtype" else qtype_onehot
        kg_cov_in = torch.zeros_like(kg_coverage) if ablation == "no_kg_coverage" else kg_coverage

        if ablation == "uniform_alpha":
            b = query_emb.shape[0]
            weights = torch.full((b, 3), 1.0 / 3.0, device=query_emb.device, dtype=query_emb.dtype)
        else:
            weights = self.controller(query_emb, qtype_in, kg_cov_in)

        if ablation == "no_kg_signal":
            denom = weights[:, 0] + weights[:, 1] + 1e-8
            a = weights[:, 0] / denom
            b = weights[:, 1] / denom
            scores = a * s_sparse + b * s_dense
            weights = torch.stack([a, b, torch.zeros_like(a)], dim=1)
            if return_breakdown:
                return scores, weights, {
                    "s_sparse": s_sparse,
                    "s_dense": s_dense,
                    "s_kg": s_kg,
                    "sparse_contrib": a * s_sparse,
                    "dense_contrib": b * s_dense,
                    "kg_contrib": torch.zeros_like(s_sparse),
                }
            return scores, weights

        alpha, beta, gamma = weights[:, 0], weights[:, 1], weights[:, 2]
        scores = alpha * s_sparse + beta * s_dense + gamma * s_kg
        if return_breakdown:
            return scores, weights, {
                "s_sparse": s_sparse,
                "s_dense": s_dense,
                "s_kg": s_kg,
                "sparse_contrib": alpha * s_sparse,
                "dense_contrib": beta * s_dense,
                "kg_contrib": gamma * s_kg,
            }
        return scores, weights
