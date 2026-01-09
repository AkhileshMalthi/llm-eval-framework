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

class ContextRelevancyMetric(BaseMetric):
    def __init__(self, provider="groq"):
        self.client = JudgeClient(provider=provider)

    def compute(self, query, response, contexts=None, reference=None) -> MetricResult:
        context_str = "\n".join(contexts if contexts else [])
        prompt = f"Query: {query}\nContext: {context_str}\nIs the context useful to answer the query? Score 1 for Yes, 0 for No. Output: SCORE: [val]"
        raw_output = self.client.ask_judge(prompt)
        
        score_match = re.search(r"SCORE:\s*([\d\.]+)", raw_output)
        score = float(score_match.group(1)) if score_match else 0.0 

        return MetricResult(name="ContextRelevancy", score=score)

class AnswerRelevancyMetric(BaseMetric):
    def __init__(self, provider="groq"):
        self.client = JudgeClient(provider=provider)

    def compute(self, query, response, contexts=None, reference=None) -> MetricResult:
        prompt = f"Query: {query}\nResponse: {response}\nDoes the response directly address the query? Score 0 to 1. Output: SCORE: [val]"
        raw_output = self.client.ask_judge(prompt)
        
        score_match = re.search(r"SCORE:\s*([\d\.]+)", raw_output)
        score = float(score_match.group(1)) if score_match else 0.0 

        return MetricResult(name="AnswerRelevancy", score=score)

# TODO: The parsing logic can extracted to a common utility function in ./utils
# TODO: The prompt templates can also be externalized for easier customization