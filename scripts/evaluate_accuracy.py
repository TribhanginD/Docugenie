"""
Evaluation harness for DocuGenie answers using retrieval-aware context and
semantic scoring.

Example usage:
    python scripts/evaluate_accuracy.py --dataset tests/eval_dataset.json

The dataset file must be JSON containing a list of objects with at least:
    {
        "question": "What is ...?",
        "reference": "Expected answer text",
        "weight": 1.0,                   # optional
        "expected_doc": "doc.pdf"        # optional, used for retrieval hit rate
    }

If a Groq or Hugging Face API key is available, the script will invoke the
configured provider to generate answers via the normal DocuGenie pipeline.
Otherwise it falls back to a deterministic stub that echoes the top chunk as
the candidate answer.
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import numpy as np
from bert_score import score as bert_score

from pages.Chatbot import (
    answer_query,
    init_retrievers,
    load_config,
    load_manual_corrections,
    process_pdfs,
    retrieve_relevant_chunks,
)
from providers.llm import GroqProvider, HFProvider, LLMProvider
try:
    from providers.llm import GeminiProvider  # type: ignore
except Exception:
    GeminiProvider = None  # type: ignore

LOGGER = logging.getLogger("docugenie.eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DocuGenie answer quality.")
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to JSON dataset with questions and reference answers.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "groq", "hf", "gemini", "stub"],
        default="auto",
        help="Which LLM provider to use for generation. 'stub' skips LLM calls.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Hybrid retrieval alpha override. Defaults to config value.",
    )
    parser.add_argument(
        "--bert-model",
        default="microsoft/deberta-base-mnli",
        help="Hugging Face model to use for BERTScore.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language hint for BERTScore.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to dump per-sample metrics as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between questions to respect rate limits (seconds).",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for i, raw in enumerate(fh):
                line = raw.strip()
                # Strip UTF-8 BOM if present on the first line
                if i == 0 and line.startswith("\ufeff"):
                    line = line.lstrip("\ufeff")
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {i+1} of {path}: {exc}"
                    ) from exc
    else:
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError("Dataset file must contain a list of question objects.")

    for idx, item in enumerate(rows):
        if "question" not in item or "reference" not in item:
            raise ValueError(f"Item {idx} missing required keys 'question'/'reference'.")
        item.setdefault("weight", 1.0)
        if "expected_doc" not in item:
            pdf_field = item.get("pdf") or item.get("document")
            if pdf_field:
                item["expected_doc"] = Path(pdf_field).name
    return rows


def resolve_provider(name: str) -> Optional[LLMProvider]:
    if name == "stub":
        LOGGER.info("Using stub provider – answers will be derived from top chunks only.")
        return None

    if name == "groq":
        try:
            LOGGER.info("Initializing GroqProvider for evaluation.")
            return GroqProvider()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("GroqProvider unavailable: %s", exc)
            return None

    if name == "hf":
        try:
            LOGGER.info("Initializing HFProvider for evaluation.")
            return HFProvider()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("HFProvider unavailable: %s", exc)
            return None

    if name == "gemini":
        try:
            if GeminiProvider is None:
                raise RuntimeError("GeminiProvider not available")
            LOGGER.info("Initializing GeminiProvider for evaluation.")
            return GeminiProvider()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("GeminiProvider unavailable: %s", exc)
            return None

    # auto mode: prefer Gemini if key present; otherwise Groq then HF
    ordered: List[Tuple[Any, str]] = []
    if GeminiProvider is not None and (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        ordered.append((GeminiProvider, "Gemini"))
    ordered.extend(((GroqProvider, "Groq"), (HFProvider, "HF")))

    for factory, label in ordered:
        try:
            LOGGER.info("Trying %s provider for evaluation.", label)
            return factory()  # type: ignore[call-arg]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("%s provider unavailable: %s", label, exc)
    LOGGER.info("Falling back to stub provider after failed auto resolution.")
    return None


def fallback_answer(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return ""
    # Join the first highlighted sentences to approximate an answer.
    texts = [chunk.get("text", "") for chunk in chunks[:2]]
    return "\n".join(text.strip() for text in texts if text.strip())


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    dataset = load_dataset(args.dataset)
    LOGGER.info("Loaded %d evaluation samples from %s", len(dataset), args.dataset)

    all_chunks, chunk_metadata, _, _ = process_pdfs()
    bm25, faiss_idx = init_retrievers(all_chunks, force_rebuild=False)
    provider = resolve_provider(args.provider)
    alpha = args.alpha if args.alpha is not None else load_config().get("alpha", 0.6)
    manual_corrections = load_manual_corrections()

    per_sample_results: List[Dict[str, Any]] = []
    candidates: List[str] = []
    references: List[str] = []

    sleep_seconds = max(0.0, args.sleep_seconds)

    for item in dataset:
        question = item["question"]
        expected_doc = item.get("expected_doc")
        LOGGER.info("Evaluating question: %s", question)

        if provider:
            try:
                answer, usage, chunks, _citations, _fact = answer_query(
                    query=question,
                    provider=provider,
                    bm25=bm25,
                    faiss_idx=faiss_idx,
                    all_chunks=all_chunks,
                    chunk_metadata=chunk_metadata,
                    memory=[],
                    corrections=manual_corrections,
                    alpha=alpha,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Generation failed (%s); falling back to stub.", exc)
                chunks = retrieve_relevant_chunks(
                    question,
                    bm25,
                    faiss_idx,
                    all_chunks,
                    chunk_metadata,
                    alpha=alpha,
                )
                corrections_text = manual_corrections.get(question, [])
                answer = "\n".join(corrections_text) if corrections_text else fallback_answer(chunks)
                usage = {"model_used": "stub"}
        else:
            chunks = retrieve_relevant_chunks(
                question,
                bm25,
                faiss_idx,
                all_chunks,
                chunk_metadata,
                alpha=alpha,
            )
            corrections_text = manual_corrections.get(question, [])
            answer = "\n".join(corrections_text) if corrections_text else fallback_answer(chunks)
            usage = {"model_used": "stub"}

        retrieved_docs = {chunk.get("doc") for chunk in chunks if chunk.get("doc")}
        hit = expected_doc in retrieved_docs if expected_doc else None

        per_sample_results.append(
            {
                "question": question,
                "answer": answer,
                "reference": item["reference"],
                "model_used": usage.get("model_used"),
                "retrieved_docs": sorted(retrieved_docs),
                "retrieval_hit": hit,
                "weight": item.get("weight", 1.0),
            }
        )
        candidates.append(answer)
        references.append(item["reference"])

        if sleep_seconds and provider is not None:
            LOGGER.debug("Sleeping %.2f seconds to respect rate limits", sleep_seconds)
            time.sleep(sleep_seconds)

    LOGGER.info("Computing BERTScore with model %s", args.bert_model)
    precision, recall, f1 = bert_score(
        candidates,
        references,
        lang=args.lang,
        model_type=args.bert_model,
        rescale_with_baseline=False,
    )
    f1_scores = f1.tolist()
    for result, f1_val in zip(per_sample_results, f1_scores):
        result["bert_score_f1"] = f1_val

    weighted_scores = [
        result["bert_score_f1"] * result.get("weight", 1.0)
        for result in per_sample_results
    ]
    total_weight = sum(result.get("weight", 1.0) for result in per_sample_results)
    macro_f1 = mean(f1_scores) if f1_scores else 0.0
    weighted_f1 = sum(weighted_scores) / total_weight if total_weight else 0.0

    retrieval_hits = [
        r["retrieval_hit"] for r in per_sample_results if r["retrieval_hit"] is not None
    ]
    retrieval_hit_rate = (
        float(np.mean(retrieval_hits)) if retrieval_hits else None
    )

    summary = {
        "samples": len(per_sample_results),
        "macro_bert_f1": macro_f1,
        "weighted_bert_f1": weighted_f1,
        "retrieval_hit_rate": retrieval_hit_rate,
    }
    LOGGER.info("Evaluation summary: %s", json.dumps(summary, indent=2))

    if args.output:
        payload = {"summary": summary, "results": per_sample_results}
        args.output.write_text(json.dumps(payload, indent=2))
        LOGGER.info("Wrote detailed report to %s", args.output)


if __name__ == "__main__":
    main()
