"""Prepare a HotpotQA-based evaluation dataset for DocuGenie.

This script downloads a small subset of the HotpotQA validation split,
converts the supporting passages into PDF files, and emits a JSONL dataset
that can be consumed by DocuGenie's evaluation harness.

Usage:
    PYTHONPATH=. python3 scripts/prepare_hotpot_eval.py \
        --count 50 \
        --split validation[:200]

Requirements:
    pip install datasets reportlab

Outputs:
    evaluation/hotpotqa/pdfs/hotpot_{i}.pdf  (per-question document)
    evaluation/hotpotqa/hotpot_eval.jsonl    (question/reference metadata)

The generated JSONL includes fields:
  • question
  • reference (ground-truth answer)
  • pdf (relative path to the generated PDF)
  • titles (supporting Wikipedia page titles)
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import List, Dict

MIN_CHAR_COUNT = 1500  # aim for at least ~one page of text per generated PDF

try:
    from datasets import load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - helpful message
    raise RuntimeError(
        "Missing dependency: datasets. Install with `pip install datasets`."
    ) from exc

try:
    from reportlab.pdfgen import canvas  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: reportlab. Install with `pip install reportlab`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PDF documents and evaluation metadata from HotpotQA"
    )
    parser.add_argument(
        "--split",
        default="validation[:50]",
        help="Dataset split slice. Defaults to first 50 validation examples.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Maximum number of examples to materialize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/hotpotqa"),
        help="Directory to store PDFs and JSONL metadata.",
    )
    return parser.parse_args()


def extract_supporting_passages(example: Dict) -> List[str]:
    """Build rich passages for PDFs including extended excerpts per title."""
    titles = example["context"].get("title", [])
    sentence_groups = example["context"].get("sentences", [])
    contexts = {
        title: (sentence_groups[idx] if idx < len(sentence_groups) else [])
        for idx, title in enumerate(titles)
    }
    supporting_raw = example.get("supporting_facts", {})
    if isinstance(supporting_raw, dict) and "title" in supporting_raw:
        titles_sf = supporting_raw.get("title", [])
        sent_ids = supporting_raw.get("sent_id", [])
        supporting = list(zip(titles_sf, sent_ids))
    else:
        supporting = list(supporting_raw)

    passages: List[str] = []
    used_titles = set()
    for fact in supporting:
        if not fact:
            continue
        title = fact[0]
        sentence_idx = fact[1] if len(fact) > 1 else None
        if isinstance(sentence_idx, str):
            if sentence_idx.isdigit():
                sentence_idx = int(sentence_idx)
            else:
                sentence_idx = None
        sentences = contexts.get(title)
        if not sentences:
            continue
        if isinstance(sentence_idx, int) and 0 <= sentence_idx < len(sentences):
            passages.append(f"{title}: {sentences[sentence_idx]}")
        used_titles.add(title)
        # Always include an extended excerpt (all sentences) to ensure >= 1 page
        if sentences:
            extended = " ".join(sentences)
            passages.append(f"{title} (extended): {extended}")
    # Fallback: if supporting facts missing, include first passage
    if not passages and titles:
        title = titles[0]
        sentences = sentence_groups[0] if sentence_groups else []
        sample = " ".join(sentences) if sentences else ""
        passages.append(f"{title}: {sample}")
        used_titles.add(title)

    # Ensure we meet minimum character count by appending additional context sections
    total_chars = sum(len(p) for p in passages)
    if total_chars < MIN_CHAR_COUNT:
        for title, sentences in zip(titles, sentence_groups):
            if total_chars >= MIN_CHAR_COUNT:
                break
            if title in used_titles or not sentences:
                continue
            extended = " ".join(sentences)
            if not extended:
                continue
            passages.append(f"{title} (extended): {extended}")
            used_titles.add(title)
            total_chars += len(extended)

    # As a final safeguard, duplicate the first passage if still below threshold
    if passages and total_chars < MIN_CHAR_COUNT:
        extra = "\n\n".join(passages)
        passages.append(extra)
    return passages


def write_pdf(path: Path, passages: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path))
    text_obj = c.beginText(40, 800)
    for section in passages:
        for line in textwrap.wrap(section, width=90):
            text_obj.textLine(line)
        text_obj.textLine("")
    c.drawText(text_obj)
    c.save()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("hotpot_qa", "distractor", split=args.split)

    out_dir = args.output_dir
    pdf_dir = out_dir / "pdfs"
    jsonl_path = out_dir / "hotpot_eval.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, str]] = []
    for idx, example in enumerate(dataset):
        if idx >= args.count:
            break
        passages = extract_supporting_passages(example)
        pdf_path = pdf_dir / f"hotpot_{idx}.pdf"
        write_pdf(pdf_path, passages)
        supporting_facts = example.get("supporting_facts", {})
        if isinstance(supporting_facts, dict):
            record_titles = supporting_facts.get("title", [])
        else:
            record_titles = [fact[0] for fact in supporting_facts]

        records.append(
            {
                "question": example["question"],
                "reference": example["answer"],
                "pdf": str(pdf_path),
                "expected_doc": pdf_path.name,
                "titles": record_titles,
            }
        )

    with jsonl_path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} HotpotQA examples to {jsonl_path}")


if __name__ == "__main__":
    main()
