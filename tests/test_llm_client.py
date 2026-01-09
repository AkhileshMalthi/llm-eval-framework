"""Test suite for LLM client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_eval.utils.llm_client import JudgeClient


class TestJudgeClient:
    """Tests for JudgeClient class."""
    
    def test_initialization_openai(self):
        """Test initialization with OpenAI provider."""
        with patch('llm_eval.utils.llm_client.OpenAI') as MockOpenAI:
            client = JudgeClient(provider="openai", model="gpt-4")
            
            assert client.provider == "openai"
            assert client.model == "gpt-4"
            MockOpenAI.assert_called_once()
    
    def test_initialization_groq(self):
        """Test initialization with Groq provider."""
        with patch('llm_eval.utils.llm_client.Groq') as MockGroq:
            client = JudgeClient(provider="groq", model="llama-3.3-70b-versatile")
            
            assert client.provider == "groq"
            assert client.model == "llama-3.3-70b-versatile"
            MockGroq.assert_called_once()
    
    def test_initialization_invalid_provider(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            JudgeClient(provider="invalid", model="some-model")
    
    def test_ask_judge_openai_success(self):
        """Test successful OpenAI API call."""
        with patch('llm_eval.utils.llm_client.OpenAI') as MockOpenAI:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Test response"
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client
            
            client = JudgeClient(provider="openai", model="gpt-4")
            result = client.ask_judge(prompt="Test prompt")
            
            assert result == "Test response"
            mock_client.chat.completions.create.assert_called_once()
    
    def test_ask_judge_groq_success(self):
        """Test successful Groq API call."""
        with patch('llm_eval.utils.llm_client.Groq') as MockGroq:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Test response from Groq"
            mock_client.chat.completions.create.return_value = mock_response
            MockGroq.return_value = mock_client
            
            client = JudgeClient(provider="groq", model="llama-3.3-70b-versatile")
            result = client.ask_judge(prompt="Test prompt")
            
            assert result == "Test response from Groq"
            mock_client.chat.completions.create.assert_called_once()
    
    def test_ask_judge_with_retry(self):
        """Test that retry logic works on failures."""
        with patch('llm_eval.utils.llm_client.OpenAI') as MockOpenAI:
            mock_client = MagicMock()
            # First call fails, second succeeds
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Success after retry"
            mock_client.chat.completions.create.side_effect = [
                Exception("API Error"),
                mock_response
            ]
            MockOpenAI.return_value = mock_client
            
            client = JudgeClient(provider="openai", model="gpt-4")
            result = client.ask_judge(prompt="Test prompt")
            
            assert result == "Success after retry"
            assert mock_client.chat.completions.create.call_count == 2
    
    def test_ask_judge_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        with patch('llm_eval.utils.llm_client.OpenAI') as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Persistent API Error")
            MockOpenAI.return_value = mock_client
            
            client = JudgeClient(provider="openai", model="gpt-4")
            
            with pytest.raises(Exception):  # Could be RetryError or the original exception
                client.ask_judge(prompt="Test prompt")
    
    def test_ask_judge_prompt_formatting(self):
        """Test that prompt is properly formatted in API call."""
        with patch('llm_eval.utils.llm_client.OpenAI') as MockOpenAI:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Response"
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client
            
            client = JudgeClient(provider="openai", model="gpt-4")
            result = client.ask_judge(prompt="Custom prompt")
            
            # Verify the method was called and returned a result
            assert mock_client.chat.completions.create.called
            assert result == "Response"
    
    def test_ask_judge_model_parameter(self):
        """Test that correct model is used in API call."""
        with patch('llm_eval.utils.llm_client.OpenAI') as MockOpenAI:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Response"
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client
            
            client = JudgeClient(provider="openai", model="gpt-4-turbo")
            client.ask_judge(prompt="Test")
            
            # Verify correct model was used
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4-turbo"
