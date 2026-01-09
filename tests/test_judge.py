"""Test suite for LLM-as-a-Judge metric."""

import pytest
from unittest.mock import Mock, patch
from llm_eval.metrics.judge import MultiDimensionalJudge
from llm_eval.metrics.base import MetricResult


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock_client = Mock()
    return mock_client


class TestMultiDimensionalJudgeMetric:
    """Tests for Multi-dimensional Judge metric."""
    
    def test_initialization_default(self):
        """Test initialization with default dimensions."""
        with patch('llm_eval.metrics.judge.JudgeClient') as MockClient:
            MockClient.return_value = Mock()
            metric = MultiDimensionalJudge(provider="openai")
            assert metric is not None
    
    def test_initialization_custom_provider(self):
        """Test initialization with custom provider."""
        with patch('llm_eval.metrics.judge.JudgeClient') as MockClient:
            MockClient.return_value = Mock()
            metric = MultiDimensionalJudge(provider="groq")
            assert metric is not None
    
    def test_compute_high_quality(self, mock_llm_client):
        """Test with high-quality response."""
        with patch('llm_eval.metrics.judge.JudgeClient', return_value=mock_llm_client):
            mock_llm_client.ask_judge.return_value = '{"coherence": 5, "relevance": 5, "safety": 5}'
            
            metric = MultiDimensionalJudge(provider="openai")
            
            query = "What is the capital of France?"
            response = "The capital of France is Paris."
            reference = "Paris is the capital of France."
            
            result = metric.compute(query=query, response=response, reference=reference)
            
            assert isinstance(result, MetricResult)
            assert result.name == "LLM-Judge"
            assert result.score == 1.0  # Perfect score: 5/5 normalized
            assert result.reasoning is not None
            assert mock_llm_client.ask_judge.called
    
    def test_compute_low_quality(self, mock_llm_client):
        """Test with low-quality response."""
        with patch('llm_eval.metrics.judge.JudgeClient', return_value=mock_llm_client):
            mock_llm_client.ask_judge.return_value = '{"coherence": 2, "relevance": 2, "safety": 2}'
            
            metric = MultiDimensionalJudge(provider="openai")
            
            query = "What is the capital of France?"
            response = "I don't know."
            reference = "Paris is the capital of France."
            
            result = metric.compute(query=query, response=response, reference=reference)
            
            assert result.score < 0.5  # Low score
    
    def test_compute_mixed_scores(self, mock_llm_client):
        """Test with mixed dimension scores."""
        with patch('llm_eval.metrics.judge.JudgeClient', return_value=mock_llm_client):
            mock_llm_client.ask_judge.return_value = '{"coherence": 4, "relevance": 3, "safety": 5}'
            
            metric = MultiDimensionalJudge(provider="openai")
            
            result = metric.compute(
                query="test",
                response="test response",
                reference="test reference"
            )
            
            assert 0.6 <= result.score <= 0.9  # Average score
            # Note: reasoning contains the breakdown
    
    def test_compute_with_contexts(self, mock_llm_client):
        """Test that contexts are properly handled."""
        with patch('llm_eval.metrics.judge.JudgeClient', return_value=mock_llm_client):
            mock_llm_client.ask_judge.return_value = '{"coherence": 4, "relevance": 4, "safety": 4}'
            
            metric = MultiDimensionalJudge(provider="openai")
            
            contexts = ["Context 1", "Context 2"]
            result = metric.compute(
                query="test",
                response="test",
                reference="test",
                contexts=contexts
            )
            
            # Verify judge was called
            assert mock_llm_client.ask_judge.called
    
    def test_compute_invalid_score_format(self, mock_llm_client):
        """Test handling of invalid score format from LLM."""
        with patch('llm_eval.metrics.judge.JudgeClient', return_value=mock_llm_client):
            mock_llm_client.ask_judge.return_value = "invalid json"
            
            metric = MultiDimensionalJudge(provider="openai")
            
            result = metric.compute(query="test", response="test", reference="test")
            
            # Should handle gracefully and return 0 or skip that dimension
            assert 0.0 <= result.score <= 1.0
