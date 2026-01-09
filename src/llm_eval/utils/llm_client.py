from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import os

class JudgeClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def ask_judge(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        if response.choices[0].message.content:
            return response.choices[0].message.content
        
        raise ValueError(f"No content in judge response\n Response: {response.to_json()}")