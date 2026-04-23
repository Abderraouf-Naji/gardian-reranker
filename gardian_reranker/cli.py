from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import GardianReranker
from .schemas import Candidate, QueryFeatures


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rerank candidates with GARDIAN.")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--config", required=True, help="Path to reranker config JSON file")
    p.add_argument("--input", required=True, help="Path to input JSON file")
    p.add_argument("--output", required=True, help="Path to output JSON file")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    return p.parse_args()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = _parse_args()
    cfg = _load_json(args.config)
    payload = _load_json(args.input)

    qf = QueryFeatures(
        query_emb=payload["query_features"]["query_emb"],
        qtype_onehot=payload["query_features"]["qtype_onehot"],
        kg_coverage=float(payload["query_features"]["kg_coverage"]),
        ablation=payload["query_features"].get("ablation"),
    )
    candidates = [
        Candidate(
            id=c["id"],
            text=c.get("text", ""),
            sparse_feats=c["sparse_feats"],
            dense_feats=c["dense_feats"],
            kg_feats=c["kg_feats"],
            metadata=c.get("metadata"),
        )
        for c in payload.get("candidates", [])
    ]

    reranker = GardianReranker.from_checkpoint(args.checkpoint, cfg, device=args.device)
    out = reranker.rerank(qf, candidates)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
