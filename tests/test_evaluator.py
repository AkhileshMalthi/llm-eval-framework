"""Test suite for the Evaluator class."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock
from llm_eval.evaluator import Evaluator
from llm_eval.metrics.base import MetricResult, BaseMetric
from llm_eval.metrics.classical import BleuMetric, RougeLMetric


@pytest.fixture
def mock_metrics():
    """Create mock metrics for testing."""
    return [BleuMetric(), RougeLMetric()]


@pytest.fixture
def sample_dataset_file():
    """Create a temporary JSONL dataset file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        data = [
            {
                "query": "What is Python?",
                "response": "Python is a programming language.",
                "expected_answer": "Python is a high-level programming language.",
                "retrieved_contexts": ["Python is a popular programming language."]
            },
            {
                "query": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "expected_answer": "Paris is the capital of France.",
                "retrieved_contexts": ["Paris is the capital city of France."]
            }
        ]
        for item in data:
            f.write(json.dumps(item) + '\n')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestEvaluator:
    """Tests for Evaluator class."""
    
    def test_initialization(self, mock_metrics):
        """Test Evaluator initialization."""
        evaluator = Evaluator(metrics=mock_metrics)
        assert len(evaluator.metrics) == 2
        assert all(isinstance(m, BaseMetric) for m in evaluator.metrics)
    
    def test_run_evaluation(self, mock_metrics, sample_dataset_file):
        """Test running evaluation on dataset."""
        evaluator = Evaluator(metrics=mock_metrics)
        
        # Run evaluation
        results_df = evaluator.run(dataset_path=sample_dataset_file)
        
        # Check results
        assert len(results_df) == 2
        assert "query" in results_df.columns
        assert "response" in results_df.columns
        assert "BLEU" in results_df.columns
        assert "ROUGE-L" in results_df.columns
    
    def test_run_with_empty_dataset(self, mock_metrics):
        """Test evaluation with empty dataset."""
        # Create empty JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            evaluator = Evaluator(metrics=mock_metrics)
            results_df = evaluator.run(dataset_path=temp_path)
            
            assert len(results_df) == 0
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_results_contain_all_fields(self, mock_metrics, sample_dataset_file):
        """Test that results contain all required fields."""
        evaluator = Evaluator(metrics=mock_metrics)
        
        results_df = evaluator.run(dataset_path=sample_dataset_file)
        
        # Check all required columns exist
        assert "query" in results_df.columns
        assert "response" in results_df.columns
        # Metric scores should be present
        assert "BLEU" in results_df.columns
        assert "ROUGE-L" in results_df.columns
        
    def test_metric_computation(self, sample_dataset_file):
        """Test that metrics are properly computed."""
        # Use real metrics to verify computation
        mock_metric = Mock(spec=BaseMetric)
        mock_metric.compute.return_value = MetricResult(
            name="TestMetric",
            score=0.85
        )
        
        evaluator = Evaluator(metrics=[mock_metric])
        results_df = evaluator.run(dataset_path=sample_dataset_file)
        
        # Verify compute was called for each row
        assert mock_metric.compute.call_count == 2
        assert "TestMetric" in results_df.columns
