"""
LLM Evaluation Framework - A production-grade evaluation suite for Large Language Models.

This package provides comprehensive evaluation metrics for LLM applications including:
- Classical NLP metrics (BLEU, ROUGE, BERTScore)
- RAG-specific metrics (Faithfulness, Context Relevancy, Answer Relevancy)
- LLM-as-a-Judge multi-dimensional evaluation

Author: Akhilesh Malthi
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Akhilesh Malthi"
__email__ = "akhileshmalthi2299@gmail.com"

# Lazy imports to keep CLI fast - imports happen only when actually used
def __getattr__(name):
    if name == "Evaluator":
        from llm_eval.evaluator import Evaluator
        return Evaluator
    elif name == "EvalConfig":
        from llm_eval.config import EvalConfig
        return EvalConfig
    elif name == "LLMJudgeConfig":
        from llm_eval.config import LLMJudgeConfig
        return LLMJudgeConfig
    elif name == "BaseMetric":
        from llm_eval.metrics.base import BaseMetric
        return BaseMetric
    elif name == "MetricResult":
        from llm_eval.metrics.base import MetricResult
        return MetricResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Evaluator", # type: ignore
    "EvalConfig", # type: ignore
    "LLMJudgeConfig", # type: ignore
    "BaseMetric", # type: ignore
    "MetricResult", # type: ignore
    "__version__",
]
