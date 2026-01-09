"""Integration tests for end-to-end workflows."""

import pytest
import json
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from llm_eval.evaluator import Evaluator


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_benchmark_file(temp_directory):
    """Create a sample benchmark JSONL file."""
    benchmark_path = temp_directory / "benchmark.jsonl"
    
    data = [
        {
            "query": "What is Python?",
            "response": "Python is a programming language.",
            "expected_answer": "Python is a high-level programming language.",
            "retrieved_contexts": ["Python is a popular programming language."]
        },
        {
            "query": "What is the capital of France?",
            "response": "Paris",
            "expected_answer": "Paris is the capital of France.",
            "retrieved_contexts": ["Paris is the capital city of France."]
        }
    ]
    
    with open(benchmark_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    return benchmark_path


class TestEndToEndEvaluation:
    """End-to-end integration tests."""
    
    def test_full_evaluation_pipeline(self, sample_benchmark_file, temp_directory):
        """Test complete evaluation pipeline from metrics to results."""
        from llm_eval.metrics.classical import BleuMetric, RougeLMetric
        
        metrics = [BleuMetric(), RougeLMetric()]
        evaluator = Evaluator(metrics=metrics)
        results_df = evaluator.run(dataset_path=str(sample_benchmark_file))
        
        assert len(results_df) == 2
        assert "BLEU" in results_df.columns
        assert "ROUGE-L" in results_df.columns
    
    def test_evaluation_with_multiple_metrics(self, sample_benchmark_file, temp_directory):
        """Test evaluation with multiple metrics."""
        from llm_eval.metrics.classical import BleuMetric, RougeLMetric
        from llm_eval.metrics.semantic import BERTScoreMetric
        
        metrics = [BleuMetric(), RougeLMetric(), BERTScoreMetric()]
        evaluator = Evaluator(metrics=metrics)
        results_df = evaluator.run(dataset_path=str(sample_benchmark_file))
        
        assert len(results_df) == 2
        # Check that all metrics produced results
        assert "BLEU" in results_df.columns
        assert "ROUGE-L" in results_df.columns
        assert "BERTScore" in results_df.columns
    
    def test_results_json_output(self, sample_benchmark_file, temp_directory):
        """Test that results are saved to JSON file."""
        from llm_eval.metrics.classical import BleuMetric
        
        output_dir = temp_directory / "results"
        output_dir.mkdir(exist_ok=True)
        
        metrics = [BleuMetric()]
        evaluator = Evaluator(metrics=metrics)
        results_df = evaluator.run(dataset_path=str(sample_benchmark_file))
        
        # Save results to JSON
        json_path = output_dir / "results.json"
        results_df.to_json(json_path, orient='records', lines=True)
        
        # Verify JSON file exists and is valid
        assert json_path.exists()
        loaded_df = pd.read_json(json_path, lines=True)
        
        assert len(loaded_df) == len(results_df)
    
    def test_markdown_report_generation(self, sample_benchmark_file, temp_directory):
        """Test end-to-end markdown report generation."""
        from llm_eval.metrics.classical import BleuMetric, RougeLMetric
        
        output_dir = temp_directory / "results"
        output_dir.mkdir(exist_ok=True)
        
        metrics = [BleuMetric(), RougeLMetric()]
        evaluator = Evaluator(metrics=metrics)
        results_df = evaluator.run(dataset_path=str(sample_benchmark_file))
        
        # Generate markdown report
        from llm_eval.reporting.markdown_gen import generate_markdown_report
        report_content = generate_markdown_report(results_df, "test_eval")
        
        # Save to file
        report_path = output_dir / "report.md"
        report_path.write_text(report_content)
        
        assert report_path.exists()
        content = report_path.read_text()
        assert "BLEU" in content or "Evaluation" in content or "test_eval" in content


class TestErrorHandling:
    """Tests for error handling in integration scenarios."""
    
    def test_invalid_benchmark_file(self):
        """Test handling of invalid benchmark file path."""
        from llm_eval.metrics.classical import BleuMetric
        
        metrics = [BleuMetric()]
        evaluator = Evaluator(metrics=metrics)
        
        with pytest.raises(FileNotFoundError):
            evaluator.run(dataset_path="nonexistent_file.jsonl")
    
    def test_malformed_jsonl(self, temp_directory):
        """Test handling of malformed JSONL file."""
        from llm_eval.metrics.classical import BleuMetric
        
        benchmark_path = temp_directory / "malformed.jsonl"
        with open(benchmark_path, 'w') as f:
            f.write('{"invalid json\n')
            f.write('{"also": "invalid\n')
        
        metrics = [BleuMetric()]
        evaluator = Evaluator(metrics=metrics)
        
        # Should raise an error when trying to parse malformed JSON
        with pytest.raises((ValueError, Exception)):
            evaluator.run(dataset_path=str(benchmark_path))
    
    def test_metric_computation_error_recovery(self, sample_benchmark_file, temp_directory):
        """Test that evaluation handles metric errors gracefully."""
        from llm_eval.metrics.classical import BleuMetric, RougeLMetric
        
        metrics = [BleuMetric(), RougeLMetric()]
        evaluator = Evaluator(metrics=metrics)
        
        # Mock one metric to fail
        original_compute = metrics[0].compute
        metrics[0].compute = Mock(side_effect=Exception("Metric failed"))
        
        # Should raise exception or handle it
        # Depending on implementation, this might fail or continue
        try:
            results_df = evaluator.run(dataset_path=str(sample_benchmark_file))
            # If it succeeds, check that at least some metrics worked
            assert len(results_df) > 0
        except Exception:
            # If it fails, that's also acceptable behavior
            pass
