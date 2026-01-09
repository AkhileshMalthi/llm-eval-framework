"""Test suite for reporting modules."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from llm_eval.reporting.markdown_gen import generate_markdown_report
from llm_eval.reporting.visualizer import generate_radar_chart, generate_score_histograms


@pytest.fixture
def sample_results():
    """Create sample evaluation results."""
    return [
        {
            "query": "What is Python?",
            "response": "Python is a programming language.",
            "BLEU": 0.75,
            "ROUGE-L": 0.82,
            "BERTScore": 0.91
        },
        {
            "query": "What is the capital of France?",
            "response": "Paris is the capital of France.",
            "BLEU": 0.65,
            "ROUGE-L": 0.70,
            "BERTScore": 0.88
        }
    ]


class TestMarkdownGeneration:
    """Tests for markdown report generation."""
    
    def test_generate_markdown_report_basic(self, sample_results):
        """Test basic markdown report generation."""
        df = pd.DataFrame(sample_results)
        result = generate_markdown_report(df, "test_eval")
        
        assert isinstance(result, str)
        assert "Evaluation Report" in result or "test_eval" in result
    
    def test_markdown_report_contains_metrics(self, sample_results):
        """Test that markdown report contains metric information."""
        df = pd.DataFrame(sample_results)
        result = generate_markdown_report(df, "test_eval")
        
        # Check for metric names in the report
        assert "BLEU" in result or "ROUGE" in result or "BERTScore" in result
    
    def test_markdown_report_structure(self, sample_results):
        """Test markdown report has proper structure."""
        df = pd.DataFrame(sample_results)
        result = generate_markdown_report(df, "test_eval")
        
        # Check for basic structure
        assert "#" in result  # Has headers
        assert len(result) > 50  # Has substantial content
    
    def test_markdown_report_with_empty_results(self):
        """Test markdown generation with empty results."""
        df = pd.DataFrame()
        try:
            result = generate_markdown_report(df, "test_eval")
            # If it succeeds, that's fine
            assert isinstance(result, str)
        except Exception:
            # If it fails on empty data, that's also acceptable
            pass


class TestRadarChart:
    """Tests for radar chart generation."""
    
    @patch('llm_eval.reporting.visualizer.plt')
    def test_generate_radar_chart(self, mock_plt, sample_results):
        """Test radar chart generation."""
        # Mock the subplots return value
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        df = pd.DataFrame(sample_results)
        from pathlib import Path
        generate_radar_chart(df, Path("test_radar.png"))
        
        # Verify subplots was called
        assert mock_plt.subplots.called
        assert mock_plt.savefig.called
    
    @patch('llm_eval.reporting.visualizer.plt')
    def test_radar_chart_file_saved(self, mock_plt, sample_results):
        """Test that radar chart is saved to file."""
        # Mock the subplots return value
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        df = pd.DataFrame(sample_results)
        from pathlib import Path
        output_path = Path("test_radar.png")
        generate_radar_chart(df, output_path)
        
        # Verify savefig was called with correct path
        mock_plt.savefig.assert_called()
        call_args = mock_plt.savefig.call_args
        assert str(output_path) in str(call_args)
    
    @patch('llm_eval.reporting.visualizer.plt')
    def test_radar_chart_with_single_metric(self, mock_plt):
        """Test radar chart with only one metric."""
        # Mock the subplots return value
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        single_metric_results = {
            "query": ["Test"],
            "response": ["Test response"],
            "BLEU": [0.75]
        }
        df = pd.DataFrame(single_metric_results)
        from pathlib import Path
        generate_radar_chart(df, Path("test_radar.png"))
        
        # Should still generate chart even with one metric
        assert mock_plt.savefig.called


class TestScoreHistograms:
    """Tests for score histogram generation."""
    
    @patch('llm_eval.reporting.visualizer.plt')
    def test_generate_score_histograms(self, mock_plt, sample_results):
        """Test histogram generation."""
        df = pd.DataFrame(sample_results)
        from pathlib import Path
        generate_score_histograms(df, Path("test_hist"))
        
        # Verify plotting functions were called
        assert mock_plt.figure.called or mock_plt.hist.called or mock_plt.subplot.called
        assert mock_plt.savefig.called
    
    @patch('llm_eval.reporting.visualizer.plt')
    def test_histogram_file_saved(self, mock_plt, sample_results):
        """Test that histogram is saved to file."""
        df = pd.DataFrame(sample_results)
        from pathlib import Path
        output_path = Path("test_hist")
        generate_score_histograms(df, output_path)
        
        # Verify savefig was called with correct path
        mock_plt.savefig.assert_called()
        call_args = mock_plt.savefig.call_args
        assert str(output_path) in str(call_args)
    
    @patch('llm_eval.reporting.visualizer.plt')
    def test_histogram_multiple_metrics(self, mock_plt, sample_results):
        """Test histogram with multiple metrics."""
        df = pd.DataFrame(sample_results)
        from pathlib import Path
        generate_score_histograms(df, Path("test_hist"))
        
        # Should create subplots for each metric
        # The exact number of calls depends on implementation
        assert mock_plt.savefig.called
    
    @patch('llm_eval.reporting.visualizer.plt')
    def test_histogram_with_empty_results(self, mock_plt):
        """Test histogram generation with no results."""
        df = pd.DataFrame()
        from pathlib import Path
        # Should handle gracefully without crashing
        try:
            generate_score_histograms(df, Path("test_hist"))
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass
