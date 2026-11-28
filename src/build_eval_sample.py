"""Phase 2 (Step 5): Build Blinded Evaluation Sheet & Private Answer Key.

Packages Phase 2 generated puns into a method-blinded, shuffled evaluation sheet
(default: 25 complete pairs x 4 prompting methods = 100 items) plus a private
answer key for Phase 3 human rating and LLM-as-a-Judge analysis.

Performs ZERO evaluation—strictly sampling, shuffling, and blinding Phase 2 output.

Example:
    python src/build_eval_sample.py \\
      --input results/generations/all_generations.csv \\
      --blinded-output results/eval/human_eval_blinded.csv \\
      --key-output results/eval/human_eval_key.csv \\
      --num-pairs 25
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from spanglish_pun_generator import VALID_METHODS
except ImportError:
    from src.spanglish_pun_generator import VALID_METHODS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "results" / "generations" / "all_generations.csv"
DEFAULT_BLINDED_CSV = PROJECT_ROOT / "results" / "eval" / "human_eval_blinded.csv"
DEFAULT_KEY_CSV = PROJECT_ROOT / "results" / "eval" / "human_eval_key.csv"

BLINDED_COLUMNS = (
    "eval_id",
    "spanish_word",
    "english_word",
    "generated_pun",
    "pun_success_0_1",
    "funniness_1_5",
    "acceptability_1_5",
    "error_notes",
)

KEY_COLUMNS = (
    "eval_id",
    "pair_id",
    "method",
    "spanish_word",
    "english_word",
    "context_word",
    "model",
    "prompt_version",
    "contains_pw",
)


def load_generations(input_path: Path) -> list[dict[str, Any]]:
    """Load Phase 2 generation records from either a flat CSV or JSONL file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Generation output file not found: {input_path}")

    records: list[dict[str, Any]] = []
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        # Deduplicate keeping latest per (pair_id, method)
        latest: dict[tuple[int, str], dict[str, Any]] = {}
        for rec in records:
            latest[(int(rec["pair_id"]), str(rec["method"]))] = rec
        return list(latest.values())

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def build_blinded_sample(
        records: list[dict[str, Any]],
        blinded_csv: Path,
        key_csv: Path,
        num_pairs: int = 25,
        seed: int = 42,
        required_methods: tuple[str, ...] = VALID_METHODS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample complete pairs across all methods, shuffle, blind, and write CSVs."""
    # Group valid, error-free records by pair_id -> method
    by_pair: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        method = str(rec.get("method", "")).strip()
        error = str(rec.get("error", "")).strip()
        pun = str(rec.get("generated_pun", "")).strip()
        if method in required_methods and not error and pun:
            by_pair[int(rec["pair_id"])][method] = rec

    complete_pair_ids = sorted(
        pid for pid, m_map in by_pair.items() if all(m in m_map for m in required_methods)
    )

    if len(complete_pair_ids) < num_pairs:
        raise ValueError(
            f"Requested {num_pairs} complete pairs, but only {len(complete_pair_ids)} "
            f"error-free pairs have all {len(required_methods)} methods present."
        )

    rng = random.Random(seed)
    sampled_pair_ids = rng.sample(complete_pair_ids, k=num_pairs)

    items: list[dict[str, Any]] = [
        by_pair[pid][method] for pid in sampled_pair_ids for method in required_methods
    ]
    rng.shuffle(items)

    pad_width = max(3, len(str(len(items))))
    blinded_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []

    for idx, rec in enumerate(items, start=1):
        eval_id = f"E{idx:0{pad_width}d}"
        blinded_rows.append(
            {
                "eval_id": eval_id,
                "spanish_word": rec.get("spanish_word", ""),
                "english_word": rec.get("english_word", ""),
                "generated_pun": rec.get("generated_pun", ""),
                "pun_success_0_1": "",
                "funniness_1_5": "",
                "acceptability_1_5": "",
                "error_notes": "",
            }
        )
        key_rows.append(
            {
                "eval_id": eval_id,
                "pair_id": int(rec["pair_id"]),
                "method": rec.get("method", ""),
                "spanish_word": rec.get("spanish_word", ""),
                "english_word": rec.get("english_word", ""),
                "context_word": rec.get("context_word", ""),
                "model": str(rec.get("model", "")),
                "prompt_version": str(rec.get("prompt_version", "")),
                "contains_pw": str(rec.get("contains_pw", "")),
            }
        )

    blinded_csv.parent.mkdir(parents=True, exist_ok=True)
    with blinded_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BLINDED_COLUMNS)
        writer.writeheader()
        writer.writerows(blinded_rows)

    key_csv.parent.mkdir(parents=True, exist_ok=True)
    with key_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_COLUMNS)
        writer.writeheader()
        writer.writerows(key_rows)

    return blinded_rows, key_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 (Step 5): Build blinded evaluation CSV + private answer key."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to Phase 2 generations CSV or JSONL.",
    )
    parser.add_argument(
        "--blinded-output",
        type=Path,
        default=DEFAULT_BLINDED_CSV,
        help="Path to write the method-blinded human evaluation CSV.",
    )
    parser.add_argument(
        "--key-output",
        type=Path,
        default=DEFAULT_KEY_CSV,
        help="Path to write the private eval_id -> (pair_id, method) answer key CSV.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=25,
        help="Number of complete pairs to sample (25 pairs x 4 methods = 100 items).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible pair sampling and shuffling (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_generations(args.input)
    blinded_rows, key_rows = build_blinded_sample(
        records=records,
        blinded_csv=args.blinded_output,
        key_csv=args.key_output,
        num_pairs=args.num_pairs,
        seed=args.seed,
    )

    method_counts = Counter(r["method"] for r in key_rows)
    print(
        f"Created blinded evaluation set: {len(blinded_rows)} items "
        f"({args.num_pairs} pairs x {len(method_counts)} methods, seed={args.seed})"
    )
    print("Method balance:", dict(sorted(method_counts.items())))
    print(f"Blinded sheet (share with raters): {args.blinded_output}")
    print(f"Private key (keep for Phase 3):    {args.key_output}")


if __name__ == "__main__":
    main()
