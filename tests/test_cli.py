import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
import yaml
import pandas as pd
from typer.testing import CliRunner

from llm_eval.cli import app, get_metric_instances
from llm_eval.config import EvalConfig, LLMJudgeConfig


runner = CliRunner()


class TestGetMetricInstances:
    """Tests for metric registry initialization."""
    
    @patch('llm_eval.utils.llm_client.OpenAI')
    @patch('llm_eval.utils.llm_client.Groq')
    def test_get_metric_instances_basic(self, mock_groq, mock_openai):
        """Test metric instances are created correctly."""
        config = EvalConfig(
            eval_name="test",
            dataset_path="test.jsonl",
            metrics=["bleu", "rouge", "bertscore"]
        )
        
        metrics = get_metric_instances(config)
        
        assert len(metrics) == 3
        assert all(hasattr(m, 'compute') for m in metrics)
    
    @patch('llm_eval.utils.llm_client.OpenAI')
    @patch('llm_eval.utils.llm_client.Groq')
    def test_get_metric_instances_with_judge(self, mock_groq, mock_openai):
        """Test metric instances with LLM judge configuration."""
        config = EvalConfig(
            eval_name="test",
            dataset_path="test.jsonl",
            metrics=["bleu", "judge"],
            llm_judge=LLMJudgeConfig(provider="groq", model="llama-3.3-70b-versatile")
        )
        
        metrics = get_metric_instances(config)
        
        assert len(metrics) == 2
    
    @patch('llm_eval.utils.llm_client.OpenAI')
    @patch('llm_eval.utils.llm_client.Groq')
    def test_get_metric_instances_all_metrics(self, mock_groq, mock_openai):
        """Test all available metrics can be instantiated."""
        config = EvalConfig(
            eval_name="test",
            dataset_path="test.jsonl",
            metrics=["bleu", "rouge", "bertscore", "faithfulness", "context_relevancy", "answer_relevancy", "judge"],
            llm_judge=LLMJudgeConfig(provider="openai", model="gpt-4")
        )
        
        metrics = get_metric_instances(config)
        
        assert len(metrics) == 7
    
    @patch('llm_eval.utils.llm_client.OpenAI')
    @patch('llm_eval.utils.llm_client.Groq')
    def test_get_metric_instances_default_provider(self, mock_groq, mock_openai):
        """Test default provider when llm_judge is None."""
        config = EvalConfig(
            eval_name="test",
            dataset_path="test.jsonl",
            metrics=["bleu"]
        )
        
        # Should not raise error even without llm_judge config
        metrics = get_metric_instances(config)
        assert len(metrics) == 1


class TestCLIRun:
    """Tests for CLI run command."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "run" in result.stdout.lower() or "Usage" in result.stdout
    
    def test_cli_run_help(self):
        """Test CLI run command help."""
        result = runner.invoke(app, ["run", "--help"])
        
        assert result.exit_code == 0
        assert "config" in result.stdout.lower()

