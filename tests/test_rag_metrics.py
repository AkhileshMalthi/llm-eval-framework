"""Test suite for RAG-specific metrics."""

import pytest
from unittest.mock import Mock, patch
from llm_eval.metrics.rag import FaithfulnessMetric, ContextRelevancyMetric, AnswerRelevancyMetric
from llm_eval.metrics.base import MetricResult


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock_client = Mock()
    mock_client.ask_judge.return_value = "8"
    return mock_client


class TestFaithfulnessMetric:
    """Tests for Faithfulness metric."""
    
    def test_initialization(self):
        """Test that Faithfulness metric initializes correctly."""
        with patch('llm_eval.metrics.rag.JudgeClient') as MockClient:
            MockClient.return_value = Mock()
            metric = FaithfulnessMetric(provider="openai", model="gpt-4")
            assert metric is not None
    
    def test_compute_high_faithfulness(self, mock_llm_client):
        """Test faithfulness with well-supported response."""
        with patch('llm_eval.metrics.rag.JudgeClient', return_value=mock_llm_client):
            metric = FaithfulnessMetric(provider="openai", model="gpt-4")
            mock_llm_client.ask_judge.return_value = "SCORE: 1"
            
            query = "What is the capital of France?"
            response = "The capital of France is Paris."
            contexts = ["Paris is the capital and most populous city of France."]
            
            result = metric.compute(query=query, response=response, contexts=contexts)
            
            assert isinstance(result, MetricResult)
            assert result.name == "Faithfulness"
            assert result.score == 1.0  # Binary: 1 means faithful
            assert mock_llm_client.ask_judge.called
    
    def test_compute_low_faithfulness(self, mock_llm_client):
        """Test faithfulness with unsupported response."""
        with patch('llm_eval.metrics.rag.JudgeClient', return_value=mock_llm_client):
            metric = FaithfulnessMetric(provider="openai", model="gpt-4")
            mock_llm_client.ask_judge.return_value = "SCORE: 0"
            
            query = "What is the capital of France?"
            response = "The capital of France is Berlin."
            contexts = ["Paris is the capital of France."]
            
            result = metric.compute(query=query, response=response, contexts=contexts)
            
            assert isinstance(result, MetricResult)
            assert result.name == "Faithfulness"
            assert result.score == 0.0
    
    def test_compute_no_contexts(self, mock_llm_client):
        """Test faithfulness with missing contexts."""
        with patch('llm_eval.metrics.rag.JudgeClient', return_value=mock_llm_client):
            metric = FaithfulnessMetric(provider="openai", model="gpt-4")
            
            result = metric.compute(query="test", response="test", contexts=None)
            
            assert result.score == 0.0
            assert not mock_llm_client.ask_judge.called


class TestContextRelevancyMetric:
    """Tests for Context Relevancy metric."""
    
    def test_initialization(self):
        """Test that Context Relevancy metric initializes correctly."""
        with patch('llm_eval.metrics.rag.JudgeClient') as MockClient:
            MockClient.return_value = Mock()
            metric = ContextRelevancyMetric(provider="groq")
            assert metric is not None
    
    def test_compute_relevant_context(self, mock_llm_client):
        """Test with highly relevant context."""
        with patch('llm_eval.metrics.rag.JudgeClient', return_value=mock_llm_client):
            metric = ContextRelevancyMetric(provider="groq")
            mock_llm_client.ask_judge.return_value = "SCORE: 9"
            
            query = "What is Python?"
            contexts = ["Python is a high-level programming language."]
            
            result = metric.compute(query=query, response="", contexts=contexts)
            
            assert isinstance(result, MetricResult)
            assert result.name == "Context Relevancy"
            assert result.score == 0.9
    
    def test_compute_irrelevant_context(self, mock_llm_client):
        """Test with irrelevant context."""
        with patch('llm_eval.metrics.rag.JudgeClient', return_value=mock_llm_client):
            metric = ContextRelevancyMetric(provider="groq")
            mock_llm_client.ask_judge.return_value = "SCORE: 2"
            
            query = "What is Python?"
            contexts = ["The Eiffel Tower is in Paris."]
            
            result = metric.compute(query=query, response="", contexts=contexts)
            
            assert result.score == 0.2
    
    def test_compute_no_contexts(self, mock_llm_client):
        """Test with missing contexts."""
        with patch('llm_eval.metrics.rag.JudgeClient', return_value=mock_llm_client):
            metric = ContextRelevancyMetric(provider="groq")
            
            result = metric.compute(query="test", response="", contexts=None)
            
            assert result.score == 0.0


class TestAnswerRelevancyMetric:
    """Tests for Answer Relevancy metric."""
    
    def test_initialization(self):
        """Test that Answer Relevancy metric initializes correctly."""
        with patch('llm_eval.metrics.rag.JudgeClient') as MockClient:
            MockClient.return_value = Mock()
            metric = AnswerRelevancyMetric(provider="groq")
            assert metric is not None
    
    def test_compute_relevant_answer(self, mock_llm_client):
        """Test with highly relevant answer."""
        with patch('llm_eval.metrics.rag.JudgeClient', return_value=mock_llm_client):
            metric = AnswerRelevancyMetric(provider="groq")
            mock_llm_client.ask_judge.return_value = "SCORE: 10"
            
            query = "What is the capital of France?"
            response = "The capital of France is Paris."
            
            result = metric.compute(query=query, response=response, contexts=[])
            
            assert isinstance(result, MetricResult)
            assert result.name == "Answer Relevancy"
            assert result.score == 1.0
    
    def test_compute_irrelevant_answer(self, mock_llm_client):
        """Test with irrelevant answer."""
        with patch('llm_eval.metrics.rag.JudgeClient', return_value=mock_llm_client):
            metric = AnswerRelevancyMetric(provider="groq")
            mock_llm_client.ask_judge.return_value = "SCORE: 1"
            
            query = "What is the capital of France?"
            response = "Python is a programming language."
            
            result = metric.compute(query=query, response=response, contexts=[])
            
            assert result.score == 0.1
    
    def test_compute_no_response(self, mock_llm_client):
        """Test with missing response."""
        with patch('llm_eval.metrics.rag.JudgeClient', return_value=mock_llm_client):
            metric = AnswerRelevancyMetric(provider="groq")
            
            result = metric.compute(query="test", response="", contexts=[])
            
            assert result.score == 0.0
