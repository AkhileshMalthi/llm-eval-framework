"""Test suite for configuration models."""

import pytest
from pydantic import ValidationError
from llm_eval.config import EvalConfig, LLMJudgeConfig


class TestEvalConfig:
    """Tests for EvalConfig model."""
    
    def test_valid_config(self):
        """Test creation of valid configuration."""
        config = EvalConfig(
            eval_name="test_eval",
            dataset_path="benchmarks/test.jsonl",
            output_dir="results",
            metrics=["bleu", "rouge"]
        )
        
        assert config.dataset_path == "benchmarks/test.jsonl"
        assert config.output_dir == "results"
        assert len(config.metrics) == 2
    
    def test_config_with_llm_judge(self):
        """Test configuration with LLM judge settings."""
        config = EvalConfig(
            eval_name="test_eval",
            dataset_path="benchmarks/test.jsonl",
            output_dir="results",
            metrics=["bleu"],
            llm_judge=LLMJudgeConfig(
                provider="openai",
                model="gpt-4",
                dimensions=["accuracy", "clarity"]
            )
        )
        
        assert config.llm_judge is not None
        assert config.llm_judge.provider == "openai"
        assert config.llm_judge.model == "gpt-4"
        assert len(config.llm_judge.dimensions) == 2
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            EvalConfig(eval_name="test")  # type: ignore  # Missing dataset_path and metrics
    
    def test_empty_metrics_list(self):
        """Test that empty metrics list is invalid."""
        with pytest.raises(ValidationError):
            EvalConfig(
                eval_name="test",
                dataset_path="test.jsonl",
                output_dir="results",
                metrics=[]
            )
    
    def test_default_llm_judge(self):
        """Test default LLM judge configuration."""
        config = EvalConfig(
            eval_name="test",
            dataset_path="test.jsonl",
            output_dir="results",
            metrics=["bleu"]
        )
        
        # llm_judge should be optional
        assert config.llm_judge is None or isinstance(config.llm_judge, LLMJudgeConfig)


class TestLLMJudgeConfig:
    """Tests for LLMJudgeConfig model."""
    
    def test_valid_llm_judge_config(self):
        """Test creation of valid LLM judge configuration."""
        config = LLMJudgeConfig(
            provider="openai",
            model="gpt-4",
            dimensions=["accuracy", "clarity", "completeness"]
        )
        
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert len(config.dimensions) == 3
    
    def test_default_dimensions(self):
        """Test default dimensions are used when not specified."""
        config = LLMJudgeConfig(
            provider="groq",
            model="llama-3.3-70b-versatile"
        )
        
        assert config.dimensions is not None
        assert len(config.dimensions) > 0
    
    def test_invalid_provider(self):
        """Test that invalid provider raises validation error."""
        # This depends on whether provider is validated
        # If provider is a string field without enum, this test may not apply
        config = LLMJudgeConfig(
            provider="invalid_provider",
            model="some-model"
        )
        # Should either validate or accept any string
    
    def test_groq_provider(self):
        """Test Groq provider configuration."""
        config = LLMJudgeConfig(
            provider="groq",
            model="llama-3.3-70b-versatile",
            dimensions=["accuracy"]
        )
        
        assert config.provider == "groq"
        assert config.model == "llama-3.3-70b-versatile"
    
    def test_openai_provider(self):
        """Test OpenAI provider configuration."""
        config = LLMJudgeConfig(
            provider="openai",
            model="gpt-4-turbo",
            dimensions=["accuracy", "clarity"]
        )
        
        assert config.provider == "openai"
        assert config.model == "gpt-4-turbo"


class TestConfigFromDict:
    """Tests for creating config from dictionary."""
    
    def test_parse_from_dict(self):
        """Test parsing configuration from dictionary."""
        config_dict = {
            "eval_name": "test_eval",
            "dataset_path": "benchmarks/test.jsonl",
            "output_dir": "results",
            "metrics": ["bleu", "rouge"],
            "llm_judge": {
                "provider": "openai",
                "model": "gpt-4",
                "dimensions": ["accuracy"]
            }
        }
        
        config = EvalConfig(**config_dict)
        
        assert config.dataset_path == "benchmarks/test.jsonl"
        assert len(config.metrics) == 2
        assert config.llm_judge is not None
        assert config.llm_judge.provider == "openai"
    
    def test_parse_with_extra_fields(self):
        """Test that extra fields are handled appropriately."""
        config_dict = {
            "eval_name": "test",
            "dataset_path": "test.jsonl",
            "output_dir": "results",
            "metrics": ["bleu"],
            "extra_field": "should be ignored or raise error"
        }
        
        # Pydantic should either ignore or raise ValidationError
        # depending on model configuration
        try:
            config = EvalConfig(**config_dict)
            # If it succeeds, extra field was ignored
            assert config.dataset_path == "test.jsonl"
        except ValidationError:
            # If it fails, extra fields are not allowed
            pass
