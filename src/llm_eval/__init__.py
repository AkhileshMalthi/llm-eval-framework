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

from llm_eval.evaluator import Evaluator
from llm_eval.config import EvalConfig, LLMJudgeConfig
from llm_eval.metrics.base import BaseMetric, MetricResult

__all__ = [
    "Evaluator",
    "EvalConfig",
    "LLMJudgeConfig",
    "BaseMetric",
    "MetricResult",
    "__version__",
]
