import re
from typing import Optional
from llm_eval.metrics.base import BaseMetric, MetricResult
from llm_eval.utils.llm_client import JudgeClient

class FaithfulnessMetric(BaseMetric):
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        self.client = JudgeClient(provider=provider, model=model)

    def compute(self, query, response, contexts=None, reference=None) -> MetricResult:
        context_str = "\n".join(contexts if contexts else [])
        prompt = f"""Verify if the response is supported by the context.
        CONTEXT: {context_str if context_str else 'N/A'}
        RESPONSE: {response}
        Output ONLY:
        SCORE: [0 or 1]
        REASON: [Short explanation]"""
        
        raw_output = self.client.ask_judge(prompt)
        
        # Simple parsing logic
        score_match = re.search(r"SCORE:\s*([\d\.]+)", raw_output)
        score = float(score_match.group(1)) if score_match else 0.0
        
        return MetricResult(name="Faithfulness", score=score, reasoning=raw_output)