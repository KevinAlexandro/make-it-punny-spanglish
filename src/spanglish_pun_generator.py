"""Gemini client wrapper for Spanglish pun generation across 4 prompting methods."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import re
import time
from typing import TYPE_CHECKING, Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional fallback before pip install
    def load_dotenv() -> bool:
        return False

if TYPE_CHECKING:
    import pandas as pd

VALID_METHODS = (
    "baseline",
    "contextually_aligned",
    "question_answer",
    "subject_masked",
)


class SpanglishPunGenerator:
    """Client wrapper for generating Spanglish puns across all 4 prompting methods."""

    # ToDo - Thinking level should be considered?
    def __init__(
            self,
            templates: dict[str, Any],
            # 2 free options:
            # gemini-4-hydrogen
            # gemini-3.5-flash-lite
            # gemini-3.8-flash
            model_id: str = "gemini-3.5-flash-lite",
            temperature: float = 0.7,
            top_p: float = 0.95,
            max_output_tokens: int = 200,
            api_key: str | None = None,
            dry_run: bool = False,
    ) -> None:
        self.templates = templates
        self.prompt_version = str(templates.get("version", "unknown"))
        self.methods = templates["methods"]
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.dry_run = dry_run
        self.client = None
        self.gen_config = None

        if not self.dry_run:
            self._init_client(api_key)

    def _init_client(self, api_key: str | None) -> None:
        """Initialize the Google GenAI client using .env / environment variables."""
        load_dotenv()
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export GEMINI_API_KEY in your shell."
            )

        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=resolved_key)
        self.gen_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
        )

    @staticmethod
    def _clean_meaning(raw_meaning: Any, max_chars: int = 60) -> str:
        return str(raw_meaning).split(";")[0].strip()[:max_chars]

    def format_prompt(
            self,
            template_key: str,
            row: pd.Series | dict[str, Any],
            masked_sentence: str = "",
    ) -> str:
        """Render the complete few-shot prompt string for a given template key and pair."""
        if template_key not in self.methods:
            raise ValueError(f"Unknown template key: {template_key}")

        tpl = self.methods[template_key]
        fields = {
            "pw": str(row["spanish_word"]).strip(),
            "pw_ipa": str(row.get("spanish_ipa", "")).strip(),
            "pw_meaning": self._clean_meaning(row["spanish_meaning"]),
            "aw": str(row["english_word"]).strip(),
            "aw_ipa": str(row.get("english_ipa", "")).strip(),
            "aw_meaning": self._clean_meaning(row["english_meaning"]),
            "cw": str(row.get("context_word", "")).strip(),
            "masked_sentence": masked_sentence.strip(),
        }

        user_query = tpl["input_template"].format(**fields)
        # Collapse any double spaces if an IPA field happens to be empty
        user_query = re.sub(r" {2,}", " ", user_query).strip()

        examples_str = "\n\n".join(
            f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in tpl["few_shot"]
        )
        return (
            f"{tpl['system']}\n\n"
            f"### Examples\n{examples_str}\n\n"
            f"### Task\nInput: {user_query}\nOutput:"
        )

    def _call_llm_with_status(
            self,
            prompt_text: str,
            max_retries: int = 4,
            base_delay: float = 2.0,
    ) -> tuple[str, str]:
        """Invoke the configured Gemini model and return (response_text, error_message)."""
        if self.dry_run or self.client is None:
            raise RuntimeError("call_llm() cannot run in dry_run mode.")

        last_error = "max retries exceeded"
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt_text,
                    config=self.gen_config,
                )
                text = (response.text or "").strip()
                return re.sub(r"\s+", " ", text), ""
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))

        return "", last_error

    def call_llm(self, prompt_text: str, max_retries: int = 4, base_delay: float = 2.0) -> str:
        """Invoke the configured Gemini model with exponential backoff retries."""
        text, error = self._call_llm_with_status(
            prompt_text, max_retries=max_retries, base_delay=base_delay
        )
        return f"[ERROR: {error}]" if error else text

    def generate_for_pair(
            self,
            row: pd.Series | dict[str, Any],
            method: str,
    ) -> dict[str, Any]:
        """Generate one Spanglish pun for a pair under the specified prompting strategy."""
        if method not in VALID_METHODS:
            raise ValueError(f"Unknown method '{method}'. Expected one of {VALID_METHODS}.")

        pw = str(row["spanish_word"]).strip()
        aw = str(row["english_word"]).strip()
        cw = str(row.get("context_word", "")).strip()
        masked_intermediate = ""
        generated_pun = ""
        error = ""

        if method in ("baseline", "contextually_aligned", "question_answer"):
            prompt = self.format_prompt(method, row)
            generated_pun, error = self._call_llm_with_status(prompt)
        else:
            # Two-stage Subject-Masked workflow
            stage1_prompt = self.format_prompt("subject_masked_stage1", row)
            masked_intermediate, stage1_err = self._call_llm_with_status(stage1_prompt)
            if stage1_err:
                error = f"stage1: {stage1_err}"
            else:
                if "[MASK]" not in masked_intermediate:
                    masked_intermediate = re.sub(
                        rf"\b{re.escape(aw)}\b",
                        "[MASK]",
                        masked_intermediate,
                        flags=re.IGNORECASE,
                    )
                stage2_prompt = self.format_prompt(
                    "subject_masked_stage2", row, masked_sentence=masked_intermediate
                )
                generated_pun, stage2_err = self._call_llm_with_status(stage2_prompt)
                if stage2_err:
                    error = f"stage2: {stage2_err}"

        contains_pw = (
            bool(re.search(rf"\b{re.escape(pw)}\b", generated_pun, flags=re.IGNORECASE))
            if (not error and generated_pun)
            else False
        )

        return {
            "pair_id": int(row["pair_id"]),
            "method": method,
            "spanish_word": pw,
            "english_word": aw,
            "context_word": cw,
            "masked_intermediate": masked_intermediate,
            "generated_pun": generated_pun,
            "model": self.model_id,
            "temperature": self.temperature,
            "prompt_version": self.prompt_version,
            "contains_pw": contains_pw,
            "error": error,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
