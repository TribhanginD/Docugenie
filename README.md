---
title: DocuGenie
emoji: 🧞
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# DocuGenie 

A production-grade, interactive document Q&A assistant powered by Retrieval-Augmented Generation (RAG).

Upload your PDFs, bring your own API key (Groq or Gemini), and ask anything.

## Features

- **Hybrid Search** — BM25 + Qdrant vector search for precision recall
- **Gemini-powered Reranking** — Listwise reranking for top-quality results
- **BYOK** — Bring Your Own API Key (Groq or Gemini) — no server costs
- **Modern React UI** — Glassmorphic design with smooth animations
- **FastAPI Backend** — Production REST API with Prometheus metrics

## Getting Started

Enter your **Groq** or **Gemini API key** in the sidebar, upload one or more PDFs, and start asking questions.

## Benchmarks

| Metric | Score |
|---|---|
| Mean Reciprocal Rank (MRR) | **1.0** |
| Precision@5 | **0.56** |
| Avg Retrieval Latency | **1.24s** |

## Tech Stack

React · Vite · TailwindCSS · FastAPI · Qdrant · Gemini · Groq

## License

MIT
