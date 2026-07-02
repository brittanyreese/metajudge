"""One-time prep: SummEval JSONL -> src/metajudge/data/demo.csv.

Usage:
    uv run python scripts/prep_demo.py --jsonl path/to/model_annotations.aligned.jsonl

The JSONL is NOT in this repo. Download from:
    https://github.com/Yale-LILY/SummEval (or Salesforce bucket)
License: MIT. See src/metajudge/data/SOURCE.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Fabbri et al. (2020) model taxonomy.
# Verify model_ids against the JSONL on first run (logged below).
SYSTEM_FAMILY: dict[str, str] = {
    "M0": "extractive",  # Lead-3
    "M1": "extractive",  # LexRank
    "M2": "abstractive",  # Seq2Seq
    "M5": "abstractive",  # Fast-RL
    "M6": "abstractive",  # rnn-ext+abs+RL+rerank
    "M8": "extractive",  # SummaRuNNer
    "M9": "abstractive",  # ML encoder-decoder
    "M10": "extractive",  # REFRESH
    "M11": "extractive",  # HBSS
    "M12": "extractive",  # GSF
    "M13": "abstractive",  # ROUGESal+Ent RL
    "M14": "abstractive",  # Bottom-Up
    "M15": "extractive",  # Ban-ETM
    "M17": "abstractive",  # Two-stage
    "M20": "abstractive",  # T5
    "M22": "abstractive",  # BART
    "M23": "abstractive",  # PEGASUS
}


def main(jsonl_path: Path, out_path: Path) -> None:
    rows: list[dict[str, object]] = []
    seen_model_ids: set[str] = set()
    id_fields_found: set[str] = set()

    with jsonl_path.open() as f:
        for line in f:
            record = json.loads(line)
            model_id: str = record["model_id"]
            seen_model_ids.add(model_id)
            doc_id: str = record.get("id", record.get("doc_id", ""))
            item = f"{doc_id}_{model_id}"
            family = SYSTEM_FAMILY.get(model_id)
            if family is None:
                print(f"WARNING: unknown model_id {model_id!r}; skipping")
                continue

            expert_annotations: list[dict[str, int]] = record["expert_annotations"]
            # R19 stability check: look for annotator ID keys
            for ann in expert_annotations:
                id_fields_found.update(
                    k for k in ann if "id" in k.lower() or "annotator" in k.lower()
                )
            for slot_idx, ann in enumerate(expert_annotations):
                rows.append(
                    {
                        "item": item,
                        "rater": f"expert_{slot_idx}",
                        "score": int(ann["coherence"]),
                        "stratum": family,
                    }
                )

    print(f"Model IDs seen in JSONL: {sorted(seen_model_ids)}")
    if id_fields_found:
        print(f"Annotator ID fields found: {id_fields_found}")
    else:
        print("R19 note: no annotator ID fields found; slot ordering assumed stable.")

    df = pd.DataFrame(rows)
    print(f"Shape: {df.shape}, items: {df['item'].nunique()}, raters: {df['rater'].nunique()}")
    print(f"Strata: {df.groupby('stratum')['item'].nunique().to_dict()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument(
        "--out",
        default=Path("src/metajudge/data/demo.csv"),
        type=Path,
    )
    args = parser.parse_args()
    main(args.jsonl, args.out)
