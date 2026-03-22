import json
import sys
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

    def test_cli_run_missing_config(self, tmp_path):
        """Test CLI exits with an error when the config file does not exist."""
        result = runner.invoke(app, ["--config", str(tmp_path / "missing.yaml")])

        assert result.exit_code != 0

    def test_cli_run_unsupported_extension(self, tmp_path):
        """Test CLI exits with an error for unsupported config file extensions."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("")

        result = runner.invoke(app, ["--config", str(config_file)])

        assert result.exit_code != 0

    def test_cli_run_loads_yaml_config(self, tmp_path):
        """Test that the CLI correctly loads a YAML config file."""
        config_data = {
            "eval_name": "yaml_test",
            "dataset_path": "benchmarks/test.jsonl",
            "metrics": ["bleu"],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        _invoke_cli_with_mocks(config_file, tmp_path)

    def test_cli_run_loads_json_config(self, tmp_path):
        """Test that the CLI correctly loads a JSON config file."""
        config_data = {
            "eval_name": "json_test",
            "dataset_path": "benchmarks/test.jsonl",
            "metrics": ["bleu"],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        _invoke_cli_with_mocks(config_file, tmp_path)


def _invoke_cli_with_mocks(config_file: Path, tmp_path: Path) -> None:
    """Invoke CLI run with all heavy dependencies mocked out and assert success."""
    mock_evaluator = MagicMock()
    mock_evaluator.run.return_value = pd.DataFrame({"bleu": [0.5]})

    mock_evaluator_module = MagicMock()
    mock_evaluator_module.Evaluator.return_value = mock_evaluator

    mock_visualizer = MagicMock()
    mock_markdown = MagicMock()
    mock_markdown.generate_markdown_report.return_value = "# Report"

    extra_modules = {
        "llm_eval.evaluator": mock_evaluator_module,
        "llm_eval.reporting.visualizer": mock_visualizer,
        "llm_eval.reporting.markdown_gen": mock_markdown,
        "dotenv": MagicMock(),
        "llm_eval.utils": MagicMock(),
        "llm_eval.utils.llm_client": MagicMock(),
        "llm_eval.metrics.classical": MagicMock(),
        "llm_eval.metrics.judge": MagicMock(),
        "llm_eval.metrics.rag": MagicMock(),
        "llm_eval.metrics.semantic": MagicMock(),
    }

    with patch.dict(sys.modules, extra_modules):
        result = runner.invoke(
            app,
            ["--config", str(config_file), "--output-dir", str(tmp_path)],
        )

    assert result.exit_code == 0
