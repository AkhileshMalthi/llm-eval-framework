# LLM Eval Framework

> **Your personal memory bank for everything LLM evaluation.**  
> Built so you never have to re-research what BLEU means again.

---

## What is this?

**llm-eval-framework** is a Python library and CLI for running reproducible, multi-metric evaluations of LLM responses. It unifies classical NLP metrics (BLEU, ROUGE), neural similarity (BERTScore), RAG-specific signals (faithfulness, context/answer relevancy), and LLM-as-Judge scoring behind a single, consistent interface.

Whether you are tuning a RAG pipeline, comparing model versions, or spot-checking a deployment, this framework gives you a structured way to measure quality—and saves the results so you can compare over time.

---

## Goals

| Goal | How the framework helps |
|------|------------------------|
| **Reproducibility** | Declarative YAML config + deterministic classical metrics |
| **Coverage** | 7 built-in metrics across 4 paradigms |
| **Speed** | Parallel evaluation via `ThreadPoolExecutor` |
| **Observability** | Radar charts, histograms, and Markdown reports auto-generated |
| **Extensibility** | One abstract class → plug in any metric in minutes |
| **Memory** | This documentation site so you remember what everything means |

---

## Feature List

- **Classical metrics** — BLEU (n-gram precision) and ROUGE-L (longest common subsequence)
- **Semantic similarity** — BERTScore using DistilBERT embeddings
- **RAG-specific metrics** — Faithfulness, Context Relevancy, Answer Relevancy (all LLM-graded)
- **LLM-as-Judge** — Multi-dimensional scoring (coherence, relevance, safety) via OpenAI or Groq
- **Parallel evaluation** — Configurable worker count; evaluates hundreds of rows in minutes
- **Rich CLI** — `llm-eval run --config config.yaml` with pretty tables and progress bars
- **Automated reporting** — Radar charts, score-distribution histograms, Markdown summaries
- **OpenAI & Groq support** — Swap providers via a single config line

---

## Quick example

```python
from llm_eval import Evaluator
from llm_eval.metrics.classical import BleuMetric, RougeLMetric

evaluator = Evaluator(metrics=[BleuMetric(), RougeLMetric()], max_workers=4)
results_df = evaluator.run("benchmarks/rag_benchmark.jsonl")
print(results_df.describe())
```

Or via the CLI:

```bash
llm-eval run --config examples/config.yaml
```

---

## Navigation guide

| You want to… | Go to… |
|---|---|
| Install and run the first evaluation | [Getting Started → Installation](getting-started/installation.md) |
| Understand what BLEU/ROUGE actually measure | [Metrics → BLEU](metrics/bleu.md) / [ROUGE](metrics/rouge.md) |
| Pick the right metric for your task | [Metrics → Overview](metrics/overview.md) |
| Understand the code structure | [Concepts → Architecture](concepts/architecture.md) |
| Add your own metric | [How-To → Add a Metric](how-to/add-a-metric.md) |
| See all CLI flags | [Reference → CLI](reference/cli.md) |
| Browse auto-generated API docs | [Reference → Python API](reference/python-api.md) |
