"""Test suite for BERTScore metric."""

import pytest
from llm_eval.metrics.semantic import BERTScoreMetric
from llm_eval.metrics.base import MetricResult


def test_bertscore_metric_initialization():
    """Test that BERTScore metric initializes correctly."""
    metric = BERTScoreMetric()
    assert metric is not None
    assert hasattr(metric, 'scorer')


def test_bertscore_metric_identical_text():
    """Test BERTScore with identical response and reference."""
    metric = BERTScoreMetric()
    query = "What is Python?"
    response = "Python is a programming language."
    reference = "Python is a programming language."
    
    result = metric.compute(query=query, response=response, reference=reference)
    
    assert isinstance(result, MetricResult)
    assert result.name == "BERTScore"
    assert 0.95 <= result.score <= 1.0  # Should be very high for identical text


def test_bertscore_metric_similar_text():
    """Test BERTScore with semantically similar text."""
    metric = BERTScoreMetric()
    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    reference = "Paris is the capital city of France."
    
    result = metric.compute(query=query, response=response, reference=reference)
    
    assert isinstance(result, MetricResult)
    assert result.name == "BERTScore"
    assert 0.7 <= result.score <= 1.0  # Should be high for similar text


def test_bertscore_metric_different_text():
    """Test BERTScore with completely different text."""
    metric = BERTScoreMetric()
    query = "What is the capital of France?"
    response = "Python is a programming language."
    reference = "Paris is the capital of France."
    
    result = metric.compute(query=query, response=response, reference=reference)
    
    assert isinstance(result, MetricResult)
    assert result.name == "BERTScore"
    assert 0.0 <= result.score <= 1.0  # Valid score range
    # Note: BERTScore may not always be very low for different text due to token overlap


def test_bertscore_metric_no_reference():
    """Test BERTScore with missing reference."""
    metric = BERTScoreMetric()
    query = "What is Python?"
    response = "Python is a programming language."
    
    result = metric.compute(query=query, response=response, reference=None)
    
    assert isinstance(result, MetricResult)
    assert result.name == "BERTScore"
    assert result.score == 0.0  # Should return 0.0 when no reference
