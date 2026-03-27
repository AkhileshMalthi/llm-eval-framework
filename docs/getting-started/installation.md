# Installation

## Requirements

- Python **3.13** or **3.14**
- An OpenAI or Groq API key (only needed for LLM-based metrics: faithfulness, context relevancy, answer relevancy, LLM-as-judge)

---

## Install from source (recommended for development)

```bash
git clone https://github.com/AkhileshMalthi/llm-eval-framework.git
cd llm-eval-framework
pip install -e .
```

This installs the `llm-eval` CLI entry point and all runtime dependencies.

---

## Install with Poetry

If you prefer [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/AkhileshMalthi/llm-eval-framework.git
cd llm-eval-framework
poetry install
```

Run commands inside the Poetry environment:

```bash
poetry run llm-eval run --config examples/config.yaml
```

---

## Docker

A `Dockerfile` and `docker-compose.yml` are included for containerised usage:

```bash
docker-compose build
docker-compose up
```

---

## Environment variables

LLM-based metrics require API keys. Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

The framework automatically loads this file via `python-dotenv`.

!!! tip "Only set what you use"
    If you only run classical metrics (BLEU, ROUGE, BERTScore), you do **not** need any API keys.

---

## Verify the installation

```bash
llm-eval --help
```

Expected output:

```
Usage: llm-eval [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  run  Run evaluation pipeline from a YAML config file.
```

---

## Runtime dependencies (summary)

| Package | Purpose |
|---------|---------|
| `typer` | CLI framework |
| `pydantic` | Config validation |
| `pyyaml` | YAML config parsing |
| `pandas` | Results DataFrame |
| `nltk` | BLEU tokenisation |
| `rouge-score` | ROUGE-L computation |
| `bert-score` + `transformers` + `torch` | BERTScore |
| `openai` / `groq` | LLM-based metrics |
| `rich` | Terminal UI |
| `matplotlib` + `seaborn` | Visualisations |
| `tenacity` | Retry logic for LLM calls |

Full pinned list: [`requirements.txt`](https://github.com/AkhileshMalthi/llm-eval-framework/blob/main/requirements.txt)
