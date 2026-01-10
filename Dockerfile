# Fast Dockerfile using UV (10-100x faster than pip/poetry)
# CPU-only optimized for LLM Evaluation Framework

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    NLTK_DATA=/app/nltk_data

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src ./src

# CRITICAL: Install PyTorch CPU-only FIRST to prevent GPU bloat
RUN uv pip install --no-cache \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies from pyproject.toml
# Use --no-deps for torch to avoid reinstalling it
RUN uv pip install --no-cache -e . --no-build-isolation

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/app/nltk_data')" || \
    python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data')"

# Create directories
RUN mkdir -p /app/results /app/benchmarks /app/examples

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import llm_eval; print('healthy')" || exit 1

CMD ["python", "-m", "llm_eval.cli", "--help"]
