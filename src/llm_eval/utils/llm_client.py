import os
from typing import Optional
from openai import OpenAI
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from llm_eval.constants import DEFAULT_LLM_PROVIDER, DEFAULT_OPENAI_MODEL, DEFAULT_GROQ_MODEL

class JudgeClient:
    def __init__(self, provider: str = DEFAULT_LLM_PROVIDER, model: Optional[str] = None):
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or DEFAULT_OPENAI_MODEL
        elif self.provider == "groq":
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.model = model or DEFAULT_GROQ_MODEL
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @retry(wait=wait_exponential(multiplier=1, min=2, max=6), stop=stop_after_attempt(3))
    def ask_judge(self, prompt: str) -> str:
        # Both Groq and OpenAI SDKs use very similar 'chat.completions.create' syntax
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        # TODO: Add response format to enforce JSON output
        if response.choices[0].message.content:
            return response.choices[0].message.content
        
        raise ValueError(f"No content in judge response\n Response: {response.to_json()}")