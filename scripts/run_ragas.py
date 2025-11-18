"""
Run evaluation on the standard dataset using DocuGenie's pipeline plus RAGAS metrics.

Requirements:
    pip install ragas datasets

Usage:
    PYTHONPATH=. python3 scripts/run_ragas.py \
        --dataset evaluation/standard_eval_dataset.jsonl \
        --output evaluation/ragas_report.json

Set one of the following to supply an LLM for RAGAS scoring:
  • OPENAI_API_KEY        (uses ragas.llms.OpenAI by default)
You can override the model via --ragas-model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from pages.Chatbot import (
    answer_query,
    init_retrievers,
    load_config,
    load_manual_corrections,
    process_pdfs,
)
from providers.llm import GroqProvider, HFProvider, LLMProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DocuGenie evaluation with RAGAS metrics.")
    parser.add_argument("--dataset", required=True, type=Path, help="JSON or JSONL dataset path.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write RAGAS results.")
    parser.add_argument(
        "--provider",
        choices=["groq", "huggingface"],
        default="groq",
        help="LLM provider for DocuGenie answers (not used by RAGAS scoring).",
    )
    parser.add_argument(
        "--ragas-model",
        default="gpt-4o-mini",
        help="Model identifier for the RAGAS-LM scorer (defaults to OpenAI gpt-4o-mini).",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        rows = json.loads(path.read_text())
    return rows


def resolve_provider(name: str) -> LLMProvider:
    if name == "groq":
        return GroqProvider()
    if name == "huggingface":
        return HFProvider()
    raise ValueError(f"Unknown provider: {name}")


def ensure_ragas_available(ragas_model: str):
    try:
        from ragas.llms import OpenAI as RagasOpenAI  # type: ignore
        from ragas.embeddings import OpenAIEmbeddings  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            AnswerRelevancy,
            Faithfulness,
            ContextPrecision,
            ContextRecall,
        )
        from datasets import Dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - informative
        raise RuntimeError(
            "Missing RAGAS dependencies. Install with `pip install ragas datasets`."
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "RAGAS scoring currently defaults to OpenAI models. "
            "Export OPENAI_API_KEY or modify scripts/run_ragas.py to plug in a different scorer."
        )

    return {
        "ragas_openai": RagasOpenAI,
        "ragas_embeddings": OpenAIEmbeddings,
        "evaluate": evaluate,
        "AnswerRelevancy": AnswerRelevancy,
        "Faithfulness": Faithfulness,
        "ContextPrecision": ContextPrecision,
        "ContextRecall": ContextRecall,
        "Dataset": Dataset,
        "ragas_model": ragas_model,
    }


def generate_records(
    dataset: List[Dict[str, Any]],
    provider: LLMProvider,
    manual_corrections: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    all_chunks, chunk_metadata, _, _ = process_pdfs()
    bm25, faiss_idx = init_retrievers(all_chunks, force_rebuild=False)
    alpha = load_config().get("alpha", 0.6)

    records: List[Dict[str, Any]] = []
    for item in dataset:
        answer, usage, chunk_results, _, _ = answer_query(
            query=item["question"],
            provider=provider,
            bm25=bm25,
            faiss_idx=faiss_idx,
            all_chunks=all_chunks,
            chunk_metadata=chunk_metadata,
            memory=[],
            corrections=manual_corrections,
            corrections_key=item["question"],
            alpha=alpha,
        )
        contexts = [chunk["text"] for chunk in chunk_results]
        records.append(
            {
                "question": item["question"],
                "contexts": contexts,
                "answer": answer,
                "ground_truth": item["reference"],
            }
        )
    return records


def main() -> None:
    args = parse_args()
    ragas_env = ensure_ragas_available(args.ragas_model)

    dataset = load_dataset(args.dataset)
    provider = resolve_provider(args.provider)
    corrections = load_manual_corrections()
    records = generate_records(dataset, provider, corrections)

    Dataset = ragas_env["Dataset"]
    ragas_dataset = Dataset.from_list(records)

    llm = ragas_env["ragas_openai"](model=args.ragas_model)
    emb = ragas_env["ragas_embeddings"]()
    metrics = [
        ragas_env["AnswerRelevancy"](),
        ragas_env["Faithfulness"](),
        ragas_env["ContextPrecision"](),
        ragas_env["ContextRecall"](),
    ]

    evaluation = ragas_env["evaluate"](ragas_dataset, metrics=metrics, llm=llm, embeddings=emb)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluation, indent=2))
    print(f"Saved RAGAS evaluation to {args.output}")


if __name__ == "__main__":
    main()
