import pytest
from llm_eval.metrics.classical import BleuMetric, RougeLMetric
from llm_eval.metrics.base import MetricResult


class TestBleuMetric:
    """Comprehensive tests for BLEU metric."""
    
    def test_bleu_metric_basic(self):
        """Test BLEU with standard inputs."""
        bleu_metric = BleuMetric()
        query = "What is the capital of France?"
        response = "The capital of France is Paris."
        contexts = []
        reference = "Paris is the capital of France."
        
        result: MetricResult = bleu_metric.compute(
            query=query, response=response, contexts=contexts, reference=reference
        )

        assert result.name == "BLEU"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.score, float)
    
    def test_bleu_missing_reference(self):
        """Test BLEU raises error when reference is missing."""
        bleu_metric = BleuMetric()
        query = "What is the capital of France?"
        response = "Paris"
        
        with pytest.raises(ValueError, match="Reference text is required"):
            bleu_metric.compute(query=query, response=response, contexts=[], reference=None)
    
    def test_bleu_identical_texts(self):
        """Test BLEU with identical reference and response."""
        bleu_metric = BleuMetric()
        text = "The capital of France is Paris."
        
        result = bleu_metric.compute(
            query="test", response=text, contexts=[], reference=text
        )
        
        # Identical texts should have high BLEU score
        assert result.score > 0.9
    
    def test_bleu_special_characters(self):
        """Test BLEU with special characters and punctuation."""
        bleu_metric = BleuMetric()
        reference = "Hello, world! How are you?"
        response = "Hello world How are you"
        
        result = bleu_metric.compute(
            query="test", response=response, contexts=[], reference=reference
        )
        
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.score, float)
    
    def test_bleu_no_overlap(self):
        """Test BLEU with completely different texts."""
        bleu_metric = BleuMetric()
        reference = "The sky is blue."
        response = "Cats are mammals."
        
        result = bleu_metric.compute(
            query="test", response=response, contexts=[], reference=reference
        )
        
        # No overlap should result in very low score
        assert result.score >= 0.0
        assert result.score < 0.3


class TestRougeLMetric:
    """Comprehensive tests for ROUGE-L metric."""
    
    def test_rouge_l_metric_basic(self):
        """Test ROUGE-L with standard inputs."""
        rouge_l_metric = RougeLMetric()
        query = "What is the capital of France?"
        response = "The capital of France is Paris."
        contexts = []
        reference = "Paris is the capital of France."
        
        result: MetricResult = rouge_l_metric.compute(
            query=query, response=response, contexts=contexts, reference=reference
        )
        
        assert result.name == "ROUGE-L"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.score, float)
    
    def test_rouge_l_missing_reference(self):
        """Test ROUGE-L raises error when reference is missing."""
        rouge_l_metric = RougeLMetric()
        query = "What is the capital of France?"
        response = "Paris"
        
        with pytest.raises(ValueError, match="Reference text is required"):
            rouge_l_metric.compute(query=query, response=response, contexts=[], reference=None)
    
    def test_rouge_l_identical_texts(self):
        """Test ROUGE-L with identical reference and response."""
        rouge_l_metric = RougeLMetric()
        text = "The capital of France is Paris."
        
        result = rouge_l_metric.compute(
            query="test", response=text, contexts=[], reference=text
        )
        
        # Identical texts should have perfect ROUGE-L score
        assert result.score == 1.0
    
    def test_rouge_l_empty_response(self):
        """Test ROUGE-L with empty response."""
        rouge_l_metric = RougeLMetric()
        
        result = rouge_l_metric.compute(
            query="test", response="", contexts=[], reference="Paris is the capital."
        )
        
        # Empty response should have zero score
        assert result.score == 0.0
    
    def test_rouge_l_partial_overlap(self):
        """Test ROUGE-L with partial text overlap."""
        rouge_l_metric = RougeLMetric()
        reference = "The quick brown fox jumps over the lazy dog."
        response = "The quick brown fox runs fast."
        
        result = rouge_l_metric.compute(
            query="test", response=response, contexts=[], reference=reference
        )
        
        # Partial overlap should give medium score
        assert 0.3 <= result.score <= 0.8
    
    def test_rouge_l_special_characters(self):
        """Test ROUGE-L with special characters."""
        rouge_l_metric = RougeLMetric()
        reference = "Hello, world! #Testing @mentions & special chars."
        response = "Hello world Testing mentions special chars"
        
        result = rouge_l_metric.compute(
            query="test", response=response, contexts=[], reference=reference
        )
        
        assert 0.0 <= result.score <= 1.0
    
    def test_rouge_l_no_overlap(self):
        """Test ROUGE-L with completely different texts."""
        rouge_l_metric = RougeLMetric()
        reference = "The sky is blue."
        response = "Cats are mammals."
        
        result = rouge_l_metric.compute(
            query="test", response=response, contexts=[], reference=reference
        )
        
        # No overlap should result in very low score
        assert result.score >= 0.0
        assert result.score < 0.2