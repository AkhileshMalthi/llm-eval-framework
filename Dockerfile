# Multi-Stage Dockerfile: CPU-only optimized for LLM Evaluation Framework
# Strategy: Install PyTorch CPU first, then filter GPU deps from requirements.txt

# ============================================================================
# STAGE 1: Builder - Install all dependencies
# ============================================================================
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=300

WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src

# STEP 1: Install PyTorch CPU-only FIRST (this prevents GPU bloat)
# When transformers/bert-score see torch is installed, they won't install GPU version
RUN uv pip install --no-cache \
    torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

# STEP 2: Install OTHER dependencies (filter out torch/nvidia since we have CPU version)
# requirements.txt has 17 GPU packages we need to skip
RUN grep -v "^torch\|^nvidia\|^triton" requirements.txt > requirements-clean.txt && \
    uv pip install --no-cache -r requirements-clean.txt

# STEP 3: Install the package itself
RUN uv pip install --no-cache --no-deps -e .

# STEP 4: Download NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/app/nltk_data')" 2>/dev/null || \
    python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data')"


# ============================================================================
# STAGE 2: Runtime - Minimal production image (only what's needed)
# ============================================================================
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NLTK_DATA=/app/nltk_data

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Copy ONLY installed packages from builder (not build tools!)
COPY --from=builder --chown=appuser:appuser /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder --chown=appuser:appuser /usr/local/bin /usr/local/bin
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/
COPY --from=builder --chown=appuser:appuser /app/nltk_data /app/nltk_data

# Create directories
RUN mkdir -p /app/results /app/benchmarks /app/examples

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import llm_eval; print('healthy')" || exit 1

CMD ["python", "-m", "llm_eval.cli", "--help"]
