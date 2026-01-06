# LLM Evaluation Framework with Multi-Metric Analysis

# Development:
## Project Structure (Planning)

```bash
llm-eval-framework/
├── pyproject.toml          # Dependencies & Metadata
├── docker-compose.yml      # One-command execution
├── src/
│   └── llm_eval/
│       ├── cli.py          # Entry point (Typer)
│       ├── config.py       # Pydantic models for YAML validation
│       ├── evaluator.py    # The "Engine" that runs the loop
│       ├── metrics/        # Base classes and specific logic
│       │   ├── base.py
│       │   ├── classical.py # (BLEU, ROUGE)
│       │   ├── semantic.py  # (BERTScore)
│       │   └── rag.py       # (Faithfulness, etc.)
│       ├── reporting/      # Markdown, JSON, and Plots
│       └── utils/          # API retries, logging
├── benchmarks/             # Your 25+ test cases (JSONL)
├── tests/                  # Pytest suite
└── .env.example            # API Key placeholders
```

