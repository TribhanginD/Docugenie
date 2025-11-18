"""
Generate a PDF briefing that captures DocuGenie's evaluation dataset and tooling.

Usage:
    python3 scripts/build_eval_brief.py

The PDF will be written to evaluation/DocuGenie_Eval_Brief.pdf and includes:
  • Dataset overview and question list
  • Steps for running the built-in BERTScore harness
  • Guidance for running RAGAS-based evaluation (optional dependency)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )
except ImportError as exc:  # pragma: no cover - helper message
    raise RuntimeError(
        "reportlab is required. Install with `pip install reportlab` before running this script."
    ) from exc

DATASET_PATH = Path("evaluation/standard_eval_dataset.jsonl")
OUTPUT_PATH = Path("evaluation/DocuGenie_Eval_Brief.pdf")


def load_dataset() -> List[Dict[str, str]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    rows: List[Dict[str, str]] = []
    with DATASET_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_table_rows(dataset: List[Dict[str, str]]) -> List[List[str]]:
    header = ["#", "Question", "Reference answer"]
    data = [header]
    for idx, row in enumerate(dataset, start=1):
        data.append(
            [
                str(idx),
                row["question"],
                row["reference"],
            ]
        )
    return data


def build_document(dataset: List[Dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=LETTER,
        title="DocuGenie Evaluation Brief",
        author="DocuGenie",
    )

    styles = getSampleStyleSheet()
    story: List = []

    story.append(Paragraph("DocuGenie Evaluation Brief", styles["Title"]))
    story.append(Spacer(1, 12))

    intro = (
        "This briefing summarizes the standard evaluation dataset and tooling used to validate "
        "DocuGenie’s retrieval-augmented generation pipeline. The dataset focuses on grounded "
        "questions drawn from the sample document “Return and replacement | Meta Store.pdf”. "
        "The toolkit section outlines how to run quantitative metrics with the project’s BERTScore "
        "harness and the optional RAGAS evaluation suite."
    )
    story.append(Paragraph(intro, styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Dataset Overview", styles["Heading2"]))
    story.append(Paragraph(
        f"Dataset file: <code>{DATASET_PATH.as_posix()}</code><br/>"
        "Entries: {}".format(len(dataset)),
        styles["BodyText"]
    ))
    story.append(Spacer(1, 8))

    table = Table(build_table_rows(dataset), colWidths=[30, 270, 240])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    story.append(table)
    story.append(PageBreak())

    story.append(Paragraph("Evaluation Toolkit", styles["Heading2"]))
    story.append(Paragraph("Built-in BERTScore Harness", styles["Heading3"]))
    story.append(Paragraph(
        "DocuGenie ships with a harness that computes BERTScore for each prompt/answer pair "
        "and reports retrieval hit rate. Run the standard evaluation dataset with:",
        styles["BodyText"]
    ))
    story.append(Paragraph(
        "<code>PYTHONPATH=. python3 scripts/evaluate_accuracy.py "
        "--dataset evaluation/standard_eval_dataset.jsonl "
        "--output evaluation/bert_score_report.json</code>",
        styles["Code"]
    ))
    story.append(Paragraph(
        "The resulting JSON report includes per-question scores, retrieval metadata, "
        "and the self-critique verdict.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("RAGAS Metrics (Optional)", styles["Heading3"]))
    story.append(Paragraph(
        "For a widely used RAG benchmark, integrate the RAGAS toolkit. "
        "Install the optional dependency and run the helper script:",
        styles["BodyText"]
    ))
    story.append(Paragraph(
        "<code>pip install ragas datasets</code><br/>"
        "<code>PYTHONPATH=. python3 scripts/run_ragas.py "
        "--dataset evaluation/standard_eval_dataset.jsonl "
        "--output evaluation/ragas_report.json</code>",
        styles["Code"]
    ))
    story.append(Paragraph(
        "The script computes Answer Relevancy, Faithfulness, Context Precision, and Context Recall "
        "using your configured LLM provider. Set the environment variables documented in the script "
        "to point at your Groq or Hugging Face endpoints.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Interpreting Results", styles["Heading3"]))
    story.append(Paragraph(
        "• Macro/weighted BERTScore ≥ 0.6 indicates strong semantic agreement.\n"
        "• RAGAS Faithfulness close to 1.0 signals that answers rely on retrieved evidence.\n"
        "• Use the self-critique verdicts to spot unsupported statements and iterate on retrieval parameters.",
        styles["BodyText"]
    ))

    doc.build(story)
    print(f"Saved evaluation brief to {OUTPUT_PATH}")


def main() -> None:
    dataset = load_dataset()
    build_document(dataset)


if __name__ == "__main__":
    main()
