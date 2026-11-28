"""Phase 2 (Steps 3 & 4): Resumable Batch Spanglish Pun Generator & Quality Summary.

Runs all 4 structured prompting strategies across a Phase 1 pun-pair CSV with
per-(pair_id, method) JSONL checkpointing/resume, automatic error retries,
deduplicated flat CSV export, and a console quality inspection report.

Example (10-pair pilot):
    python src/generate_puns.py \\
      --input data/sample_spanish_english_pun_pairs.csv \\
      --prompts prompts/templates.json \\
      --output results/generations/pilot_10.jsonl \\
      --limit 10
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional fallback before pip install
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable

try:
    from phase_2 import (
        DEFAULT_INPUT_CSV,
        DEFAULT_PROMPTS_JSON,
        PROJECT_ROOT,
        check_few_shot_leakage,
        load_prompt_templates,
        load_pun_pairs,
    )
    from spanglish_pun_generator import VALID_METHODS, SpanglishPunGenerator
except ImportError:
    from src.phase_2 import (
        DEFAULT_INPUT_CSV,
        DEFAULT_PROMPTS_JSON,
        PROJECT_ROOT,
        check_few_shot_leakage,
        load_prompt_templates,
        load_pun_pairs,
    )
    from src.spanglish_pun_generator import VALID_METHODS, SpanglishPunGenerator

DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "results" / "generations" / "generations.jsonl"

OUTPUT_COLUMNS = (
    "pair_id",
    "method",
    "spanish_word",
    "english_word",
    "context_word",
    "masked_intermediate",
    "generated_pun",
    "model",
    "temperature",
    "prompt_version",
    "contains_pw",
    "error",
    "timestamp_utc",
)

FILLER_PREFIX_RE = re.compile(
    r"^\s*(here(?:'s| is)|sure[,!:]|certainly[,!:]|of course[,!:]|pun:)",
    flags=re.IGNORECASE,
)


def read_jsonl_records(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read all valid JSON objects from a JSONL checkpoint file."""
    if not jsonl_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"[WARNING] Skipping malformed JSONL line {line_num} in {jsonl_path}: {exc}"
                )
    return records


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest record per (pair_id, method), sorted cleanly for export."""
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for rec in records:
        key = (int(rec["pair_id"]), str(rec["method"]))
        latest[key] = rec

    method_rank = {m: idx for idx, m in enumerate(VALID_METHODS)}
    return sorted(
        latest.values(),
        key=lambda r: (int(r["pair_id"]), method_rank.get(str(r["method"]), 999)),
    )


def load_completed_keys(
        jsonl_path: Path,
        retry_errors: bool = True,
) -> set[tuple[int, str]]:
    """Return completed (pair_id, method) keys from an existing JSONL checkpoint.

    When `retry_errors=True`, keys whose latest record has a non-empty `error`
    are NOT treated as completed so they will automatically be retried on resume.
    """
    deduped = deduplicate_records(read_jsonl_records(jsonl_path))
    completed: set[tuple[int, str]] = set()
    for rec in deduped:
        key = (int(rec["pair_id"]), str(rec["method"]))
        has_error = bool(str(rec.get("error", "")).strip())
        if not retry_errors or not has_error:
            completed.add(key)
    return completed


def append_jsonl_record(jsonl_path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record and flush immediately for crash-safe resume."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def export_jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> list[dict[str, Any]]:
    """Export a deduplicated, flat UTF-8 CSV summary from the JSONL checkpoint."""
    deduped = deduplicate_records(read_jsonl_records(jsonl_path))
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in deduped:
            writer.writerow({col: rec.get(col, "") for col in OUTPUT_COLUMNS})

    return deduped


def print_quality_summary(records: list[dict[str, Any]], preview_pairs: int = 3) -> None:
    """Print per-method quality metrics, format anomaly checks, and a sample preview."""
    if not records:
        print("\nNo generation records available for quality summary.")
        return

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_method[str(rec["method"])].append(rec)

    print("\n" + "=" * 78)
    print("BATCH QUALITY SUMMARY")
    print("=" * 78)
    header = (
        f"{'Method':<22} {'Total':>5} {'OK':>5} {'Err':>4} "
        f"{'Has Pw %':>9} {'Avg Wds':>8} {'Anomalies':>10}"
    )
    print(header)
    print("-" * 78)

    for method in VALID_METHODS:
        m_recs = by_method.get(method, [])
        if not m_recs:
            continue
        total = len(m_recs)
        ok_recs = [r for r in m_recs if not str(r.get("error", "")).strip()]
        ok_count = len(ok_recs)
        err_count = total - ok_count
        pw_hits = sum(1 for r in ok_recs if bool(r.get("contains_pw")))
        pw_pct = (100.0 * pw_hits / ok_count) if ok_count else 0.0
        avg_words = (
            sum(len(str(r.get("generated_pun", "")).split()) for r in ok_recs) / ok_count
            if ok_count
            else 0.0
        )

        anomalies = 0
        for r in ok_recs:
            pun = str(r.get("generated_pun", "")).strip()
            if not pun or FILLER_PREFIX_RE.search(pun):
                anomalies += 1
            elif method == "question_answer" and ("Q:" not in pun or "A:" not in pun):
                anomalies += 1
            elif method == "subject_masked" and "[MASK]" not in str(
                    r.get("masked_intermediate", "")
            ):
                anomalies += 1

        print(
            f"{method:<22} {total:>5} {ok_count:>5} {err_count:>4} "
            f"{pw_pct:>8.1f}% {avg_words:>8.1f} {anomalies:>10}"
        )

    # Side-by-side preview of the first few pairs
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_pair[int(rec["pair_id"])].append(rec)

    preview_ids = sorted(by_pair)[:preview_pairs]
    if preview_ids:
        print("\n" + "=" * 78)
        print(f"SAMPLE PREVIEW (first {len(preview_ids)} pairs)")
        print("=" * 78)
        for pid in preview_ids:
            p_recs = by_pair[pid]
            first = p_recs[0]
            print(f"\nPair #{pid}: {first['spanish_word']} <-> {first['english_word']}")
            for r in p_recs:
                if r.get("error"):
                    print(f"  [{r['method']}] ERROR: {r['error']}")
                else:
                    pw_tag = "pw=OK" if r.get("contains_pw") else "pw=MISS"
                    print(f"  [{r['method']} | {pw_tag}] {r['generated_pun']}")


def run_batch(
        rows: list[dict[str, Any]],
        methods: tuple[str, ...],
        generator: SpanglishPunGenerator,
        output_jsonl: Path,
        csv_output: Path,
        delay: float = 1.0,
        retry_errors: bool = True,
        dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Execute resumable batch pun generation and export the deduplicated CSV."""
    completed_keys = load_completed_keys(output_jsonl, retry_errors=retry_errors)
    tasks = [
        (row, method)
        for row in rows
        for method in methods
        if (int(row["pair_id"]), method) not in completed_keys
    ]

    total_planned = len(rows) * len(methods)
    skipped = total_planned - len(tasks)
    print(
        f"Batch plan: {total_planned} total ({len(rows)} pairs x {len(methods)} methods) | "
        f"Already completed: {skipped} | Remaining: {len(tasks)}"
    )

    if dry_run:
        print("[DRY-RUN] No API calls executed and no files written.")
        return deduplicate_records(read_jsonl_records(output_jsonl))

    completed_now = 0
    try:
        for idx, (row, method) in enumerate(
                tqdm(tasks, desc="Generating puns", unit="pun")
        ):
            record = generator.generate_for_pair(row, method)
            append_jsonl_record(output_jsonl, record)
            completed_now += 1

            if delay > 0 and idx < len(tasks) - 1:
                time.sleep(delay)
    except KeyboardInterrupt:
        print(
            f"\n[INTERRUPTED] Saved {completed_now} new records to {output_jsonl}. "
            "Re-run the same command to resume."
        )

    deduped = export_jsonl_to_csv(output_jsonl, csv_output)
    print(f"\nCheckpoint JSONL: {output_jsonl}")
    print(f"Flat summary CSV: {csv_output} ({len(deduped)} deduplicated rows)")
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Resumable batch Spanglish pun generator with CSV export."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to the Phase 1 pun pairs CSV.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_JSON,
        help="Path to the versioned JSON prompt templates file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Path to the resumable output JSONL checkpoint file.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Path to flat summary CSV (defaults to --output with .csv suffix).",
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
        default=None,
        help="Optional cap on number of pairs (e.g., --limit 10 for pilot).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional Gemini model override (defaults to SpanglishPunGenerator default).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature locked across all methods (default: 0.7).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to pause between generations for free-tier rate limits (default: 1.0).",
    )
    parser.add_argument(
        "--exclude-leaks",
        action="store_true",
        help="Exclude any input pairs that overlap with held-out few-shot examples.",
    )
    parser.add_argument(
        "--no-retry-errors",
        action="store_true",
        help="Treat previously errored (pair_id, method) entries as finished on resume.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check dataset, leakage, and remaining checkpoint tasks without calling the API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_output = args.csv_output or args.output.with_suffix(".csv")

    templates = load_prompt_templates(args.prompts)
    df = load_pun_pairs(args.input)
    df, leaks = check_few_shot_leakage(df, templates, exclude_leaks=args.exclude_leaks)

    print(
        f"Loaded {len(df)} pairs from {args.input} | "
        f"Templates v{templates.get('version')} ({args.prompts.name})"
    )
    if leaks:
        action = "excluded" if args.exclude_leaks else "detected (use --exclude-leaks to drop)"
        print(f"[WARNING] Few-shot leakage {action}: {leaks}")
    else:
        print("Leakage check passed: 0 held-out few-shot pairs appear in the dataset.")

    if args.limit is not None:
        df = df.head(args.limit)

    methods_to_run = VALID_METHODS if args.method == "all" else (args.method,)
    gen_kwargs: dict[str, Any] = {
        "templates": templates,
        "temperature": args.temperature,
        "dry_run": args.dry_run,
    }
    if args.model:
        gen_kwargs["model_id"] = args.model

    generator = SpanglishPunGenerator(**gen_kwargs)
    rows = df.to_dict(orient="records")

    deduped = run_batch(
        rows=rows,
        methods=methods_to_run,
        generator=generator,
        output_jsonl=args.output,
        csv_output=csv_output,
        delay=args.delay,
        retry_errors=not args.no_retry_errors,
        dry_run=args.dry_run,
    )
    if deduped:
        print_quality_summary(deduped)


if __name__ == "__main__":
    main()
