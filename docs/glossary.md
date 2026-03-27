# Glossary

Definitions for terms used throughout the framework and its documentation.

---

## A

**Abstractive summarisation**  
A summarisation approach where the model generates new sentences rather than copying from the source. Contrasted with *extractive summarisation*. BLEU/ROUGE are unreliable for abstractive outputs because they penalise paraphrases.

**Answer Relevancy**  
A RAG metric that measures whether a generated response actually addresses the query. Graded by an LLM on a 0–10 scale, normalised to 0–1. Does not check factual accuracy.

---

## B

**BaseMetric**  
The abstract base class all metrics in this framework must inherit from. Defines the `compute(query, response, contexts, reference) → MetricResult` interface.

**Benchmark**  
In this framework, a JSONL file containing evaluation samples. Each sample is a JSON object with fields like `query`, `response`, `expected_answer`, and `retrieved_contexts`.

**BERTScore**  
A semantic similarity metric that compares token-level contextual embeddings between a candidate and reference using a pre-trained transformer model. Returns precision, recall, and F1; this framework uses F1.

**BLEU (Bilingual Evaluation Understudy)**  
An n-gram precision metric comparing a candidate text to one or more reference texts. Originally designed for machine translation. Penalises candidates shorter than the reference via a brevity penalty.

**Brevity Penalty**  
A multiplier in BLEU that penalises candidate texts shorter than the reference, preventing trivially short outputs from scoring artificially high on precision.

---

## C

**Candidate**  
The generated text being evaluated — i.e. the LLM's response. Contrasted with the *reference*.

**Coherence**  
One of the three dimensions in the LLM-as-Judge metric. Measures whether the response is logically structured and internally consistent.

**Context Relevancy**  
A RAG metric measuring how relevant the retrieved context documents are to the query. Graded by an LLM on a 0–10 scale, normalised to 0–1.

---

## D

**DistilBERT**  
A smaller, faster distilled version of BERT used as the default embedding model for BERTScore in this framework.

---

## E

**Embeddings**  
Dense vector representations of text produced by a neural network. Used by BERTScore to compute semantic similarity via cosine distance.

**EvalConfig**  
The Pydantic model that validates the YAML configuration file. Contains eval name, dataset path, metric list, output directory, and LLM judge settings.

**Evaluator**  
The core class (`llm_eval.evaluator.Evaluator`) that orchestrates parallel metric computation over a JSONL dataset, returning a pandas DataFrame of scores.

---

## F

**F-measure (F1 score)**  
The harmonic mean of precision and recall: `2 × P × R / (P + R)`. Used in ROUGE-L (balances coverage and conciseness) and BERTScore.

**Faithfulness**  
A RAG metric measuring whether the response is supported by the retrieved contexts. Binary (0 or 1), graded by an LLM. Low faithfulness indicates hallucination.

---

## G

**Groq**  
An inference provider offering fast LLM inference. Supported as an alternative to OpenAI for LLM-based metrics. Set via `llm_judge.provider: "groq"` in the config.

---

## H

**Hallucination**  
When a model generates content not grounded in its input context or factual reality. Detected by the Faithfulness metric in RAG pipelines.

---

## J

**JSONL (JSON Lines)**  
A text format where each line is a valid JSON object. Used as the dataset format for this framework because it is streamable and self-describing per row.

**JudgeClient**  
The internal class (`llm_eval.utils.llm_client.JudgeClient`) that wraps OpenAI and Groq API clients with a uniform `ask_judge(prompt)` interface and automatic retry logic.

---

## L

**LCS (Longest Common Subsequence)**  
The basis of ROUGE-L. The longest sequence of tokens that appears in both the candidate and reference in the same order, though not necessarily contiguously.

**LLM-as-Judge**  
An evaluation paradigm where a large language model is prompted to rate the quality of another model's output. This framework implements it as `MultiDimensionalJudge`, scoring coherence, relevance, and safety on a 1–5 scale.

**LLMJudgeConfig**  
Pydantic model for LLM judge settings: `provider`, `model`, and `dimensions`.

---

## M

**MetricResult**  
Pydantic model returned by every metric's `compute()` call. Contains `name` (string), `score` (float, always 0–1), and optional `reasoning` (string, for LLM-based metrics).

**MkDocs**  
The documentation site generator used to build this documentation site. Uses Markdown files and a `mkdocs.yml` configuration.

**MultiDimensionalJudge**  
The LLM-as-Judge metric class. Rates responses on coherence, relevance, and safety using a structured JSON prompt, then averages and normalises the scores.

---

## N

**N-gram**  
A contiguous sequence of *n* tokens. BLEU uses 1-grams (unigrams) through 4-grams (4-grams) for its precision calculation.

**Normalisation**  
The process of scaling raw metric scores to the [0, 1] range. LLM-based metrics in this framework use normalisation (dividing by the max possible score) so all metrics are comparable on the same scale.

---

## P

**Precision**  
The fraction of predicted tokens that appear in the reference. BLEU is primarily a precision metric.

---

## R

**RAG (Retrieval-Augmented Generation)**  
An LLM architecture pattern where relevant documents are retrieved from a knowledge base and provided as context to the generator. The three RAG metrics in this framework evaluate faithfulness, context relevancy, and answer relevancy.

**Recall**  
The fraction of reference tokens that appear in the prediction. ROUGE is primarily a recall metric.

**Reference**  
A ground-truth text used as the target for reference-based metrics (BLEU, ROUGE-L, BERTScore). Provided via the `expected_answer` field in the JSONL dataset.

**Relevance**  
One of the three dimensions in the LLM-as-Judge metric. Measures whether the response actually answers the query.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**  
A family of recall-focused metrics for summarisation. ROUGE-L (longest common subsequence) is the variant implemented in this framework.

---

## S

**Safety**  
One of the three dimensions in the LLM-as-Judge metric. Measures whether the response is free of harmful, biased, or inappropriate content.

**Semantic similarity**  
The degree to which two texts convey the same meaning, regardless of surface-form differences. Measured by BERTScore using embedding cosine similarity.

**Smoothing (BLEU)**  
A technique applied to BLEU to avoid zero scores when higher-order n-grams have zero matches (common in short sentences). This framework uses `SmoothingFunction().method1` from NLTK.

**Stemming**  
Reducing words to their root form (e.g. "running" → "run"). Used by ROUGE-L (`use_stemmer=True`) to improve token matching.

---

## T

**tenacity**  
The retry library used by `JudgeClient` to automatically retry failed LLM API calls with exponential back-off (min 2s, max 6s, max 3 attempts).

**ThreadPoolExecutor**  
Python's built-in thread pool used by `Evaluator` for parallel metric computation. Configured via `max_workers` in the config.

---

## W

**Worker**  
A thread in the `ThreadPoolExecutor`. Configured via `max_workers` (default: 4). For LLM-based metrics, more workers increase throughput up to the API rate limit.
