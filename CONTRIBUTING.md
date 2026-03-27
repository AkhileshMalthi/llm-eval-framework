# Contributing

Thank you for your interest in contributing to **llm-eval-framework**!

---

## Working on the documentation locally

The documentation site is built with [MkDocs](https://www.mkdocs.org/) and the [Material theme](https://squidfunk.github.io/mkdocs-material/).

### 1. Create and activate a virtual environment

```bash
# Use the Python 3.13 executable (adjust if it's named differently on your system)
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 2. Install documentation dependencies

```bash
pip install -r docs/requirements.txt
```

This installs:

- `mkdocs` — static site generator
- `mkdocs-material` — Material theme
- `mkdocstrings[python]` — auto API docs from docstrings

### 3. Install the package (required for API docs generation)

```bash
pip install -e . --no-deps
```

The `--no-deps` flag skips heavy ML dependencies (torch, transformers, etc.) that are not needed to *build* the docs.

### 4. Start the live-preview server

```bash
mkdocs serve
```

Open [http://127.0.0.1:8000/llm-eval-framework/](http://127.0.0.1:8000/llm-eval-framework/) in your browser.  
The site auto-reloads whenever you save a Markdown file.

### 5. Build a static copy (optional)

```bash
mkdocs build
```

Output goes to `site/` (git-ignored).

---

## Documentation structure

```
docs/
├── index.md                    # Home page
├── requirements.txt            # Doc build dependencies
├── getting-started/
│   ├── installation.md
│   └── quickstart.md
├── concepts/
│   ├── architecture.md
│   └── data-model.md
├── metrics/
│   ├── overview.md             # Cheat-sheet table
│   ├── bleu.md
│   ├── rouge.md
│   ├── bertscore.md
│   ├── rag.md
│   └── llm-judge.md
├── how-to/
│   └── add-a-metric.md
├── reference/
│   ├── cli.md
│   └── python-api.md           # mkdocstrings directives
├── faq.md
└── glossary.md
```

The navigation is defined in `mkdocs.yml` at the root of the repository.

---

## Working on the code

### Install development dependencies

```bash
pip install -e ".[dev]"
# or with poetry:
poetry install
```

### Run tests

```bash
pytest
```

### Run tests with coverage

```bash
pytest --cov=llm_eval --cov-report=term-missing
```

---

## Deployment

The documentation site is automatically deployed to GitHub Pages when changes are pushed to the `main` branch, via the workflow defined in `.github/workflows/docs.yml`.

The live site is available at:  
**<https://akhileshmalthi.github.io/llm-eval-framework/>**
