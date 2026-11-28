"""Phase 2: Structured Spanglish Pun Generation Pipeline.

Loads Spanish-English pun pairs and versioned JSON prompt templates
(Asapu et al., 2025 + Attardo's GTVH framing) and generates cross-lingual
Spanglish puns via the Google Gemini API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from spanglish_pun_generator import VALID_METHODS, SpanglishPunGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "sample_spanish_english_pun_pairs.csv"
DEFAULT_PROMPTS_JSON = PROJECT_ROOT / "prompts" / "templates.json"

REQUIRED_TEMPLATE_KEYS = (
    "baseline",
    "contextually_aligned",
    "question_answer",
    "subject_masked_stage1",
    "subject_masked_stage2",
)

NON_WORD_PATTERN = r"proper name|informal name-like|phrase-like form"


def load_prompt_templates(json_path: str | Path = DEFAULT_PROMPTS_JSON) -> dict[str, Any]:
    """Load and validate the versioned JSON prompt templates."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt templates file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "methods" not in data or not isinstance(data["methods"], dict):
        raise ValueError(f"Invalid templates schema in {path}: missing 'methods' object.")

    missing = [k for k in REQUIRED_TEMPLATE_KEYS if k not in data["methods"]]
    if missing:
        raise ValueError(f"Missing required method templates in {path}: {missing}")

    return data


def load_pun_pairs(
        csv_path: str | Path = DEFAULT_INPUT_CSV,
        filter_invalid: bool = True,
) -> pd.DataFrame:
    """Load pun pairs with schema tolerance across raw seed and Phase 1 CSVs.

    Handles both `sample_spanish_english_pun_pairs.csv` (no pair_id / context_word)
    and finalized Phase 1 exports (`spanglish_pun_pairs_130_clean.csv` / `_final.csv`).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Pun pairs CSV not found: {path}")

    df = pd.read_csv(path)
    required_cols = {"spanish_word", "english_word", "spanish_meaning", "english_meaning"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV {path} is missing required columns: {sorted(missing_cols)}")

    if filter_invalid and "english_meaning" in df.columns:
        mask_valid = ~df["english_meaning"].astype(str).str.contains(
            NON_WORD_PATTERN, case=False, na=False
        )
        df = df[mask_valid].copy()

    df = df.reset_index(drop=True)

    # Ensure optional IPA columns exist
    for ipa_col in ("spanish_ipa", "english_ipa"):
        if ipa_col not in df.columns:
            df[ipa_col] = ""
        else:
            df[ipa_col] = df[ipa_col].fillna("").astype(str).str.strip()

    # Auto-assign pair_id if absent (e.g., raw seed CSV)
    if "pair_id" not in df.columns:
        df.insert(0, "pair_id", range(1, len(df) + 1))

    # Fallback context_word to english_meaning if Phase 1 Cw column is not yet present
    if "context_word" not in df.columns:
        df["context_word"] = df["english_meaning"].fillna("").astype(str).str.strip()
    else:
        df["context_word"] = df["context_word"].fillna(df["english_meaning"]).astype(str).str.strip()

    return df


def check_few_shot_leakage(
        df: pd.DataFrame,
        templates: dict[str, Any],
        exclude_leaks: bool = False,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Verify that held-out few-shot examples do not appear in the evaluation CSV."""
    held_out = templates.get("held_out_pairs", [])
    es_set = set(df["spanish_word"].astype(str).str.strip().str.lower())
    en_set = set(df["english_word"].astype(str).str.strip().str.lower())

    leaks: list[tuple[str, str]] = []
    leak_es: set[str] = set()
    leak_en: set[str] = set()

    for pair in held_out:
        es_w = str(pair.get("spanish_word", "")).strip().lower()
        en_w = str(pair.get("english_word", "")).strip().lower()
        if (es_w and es_w in es_set) or (en_w and en_w in en_set):
            leaks.append((es_w, en_w))
            leak_es.add(es_w)
            leak_en.add(en_w)

    if leaks and exclude_leaks:
        keep_mask = ~(
                df["spanish_word"].astype(str).str.strip().str.lower().isin(leak_es)
                | df["english_word"].astype(str).str.strip().str.lower().isin(leak_en)
        )
        df = df[keep_mask].reset_index(drop=True)

    return df, leaks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Spanglish pun generator across 4 structured prompting methods."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to the input Spanish-English pun pairs CSV.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_JSON,
        help="Path to the versioned JSON prompt templates file.",
    )
    parser.add_argument(
        "--method",
        choices=(*VALID_METHODS, "all"),
        default="all",
        help="Prompting strategy to run (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of pairs to preview/run (default: 1 for quick smoke test).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Format and print prompts offline without calling the Gemini API.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable filtering of phrase-like / proper-name rows from raw seed CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    templates = load_prompt_templates(args.prompts)
    df = load_pun_pairs(args.input, filter_invalid=not args.no_filter)
    df, leaks = check_few_shot_leakage(df, templates, exclude_leaks=False)

    print(
        f"Loaded {len(df)} pairs from {args.input.name} | "
        f"Templates v{templates.get('version')} ({args.prompts.name})"
    )
    if leaks:
        print(f"[WARNING] Few-shot data leakage detected for pairs: {leaks}")
    else:
        print("Leakage check passed: 0 held-out few-shot pairs appear in the dataset.")

    methods_to_run = VALID_METHODS if args.method == "all" else (args.method,)
    generator = SpanglishPunGenerator(templates=templates, dry_run=args.dry_run)
    subset = df.head(args.limit)

    for _, row in subset.iterrows():
        print(f"\n{'=' * 72}")
        print(
            f"Pair #{row['pair_id']}: Pw='{row['spanish_word']}' {row['spanish_ipa']} "
            f"<-> Aw='{row['english_word']}' {row['english_ipa']}"
        )
        print(f"{'=' * 72}")

        for method in methods_to_run:
            if args.dry_run:
                if method == "subject_masked":
                    p1 = generator.format_prompt("subject_masked_stage1", row)
                    p2 = generator.format_prompt(
                        "subject_masked_stage2",
                        row,
                        masked_sentence=f"<Stage 1 sentence with [MASK] for '{row['english_word']}'>",
                    )
                    print(f"\n--- [{method} :: stage 1] ---\n{p1}")
                    print(f"\n--- [{method} :: stage 2] ---\n{p2}")
                else:
                    prompt = generator.format_prompt(method, row)
                    print(f"\n--- [{method}] ---\n{prompt}")
            else:
                result = generator.generate_for_pair(row, method)
                print(f"\n--- [{method}] (contains_pw={result['contains_pw']}) ---")
                if result["masked_sentence"]:
                    print(f"Stage 1 Masked: {result['masked_sentence']}")
                print(f"Pun: {result['generated_pun']}")


if __name__ == "__main__":
    main()
