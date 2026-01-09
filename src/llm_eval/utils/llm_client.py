import os
from typing import Optional
from openai import OpenAI
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

class JudgeClient:
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4o-mini"
        elif self.provider == "groq":
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.model = model or "llama-3.3-70b-versatile"
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
        if response.choices[0].message.content:
            return response.choices[0].message.content
        
        raise ValueError(f"No content in judge response\n Response: {response.to_json()}")