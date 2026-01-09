import re
from typing import Optional
from llm_eval.metrics.base import BaseMetric, MetricResult
from llm_eval.utils.llm_client import JudgeClient

class FaithfulnessMetric(BaseMetric):
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        self.client = JudgeClient(provider=provider, model=model)

    def compute(self, query, response, contexts=None, reference=None) -> MetricResult:
        if not contexts:
            return MetricResult(name="Faithfulness", score=0.0, reasoning="No contexts provided")
        
        context_str = "\n".join(contexts)
        prompt = f"""Verify if the response is supported by the context.
        CONTEXT: {context_str}
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
        if not contexts:
            return MetricResult(name="Context Relevancy", score=0.0)
        
        context_str = "\n".join(contexts)
        prompt = f"Query: {query}\nContext: {context_str}\nRate how useful this context is to answer the query. Score 0-10. Output: SCORE: [val]"
        raw_output = self.client.ask_judge(prompt)
        
        score_match = re.search(r"SCORE:\s*([\d\.]+)", raw_output)
        score = float(score_match.group(1)) / 10.0 if score_match else 0.0 

        return MetricResult(name="Context Relevancy", score=score)

class AnswerRelevancyMetric(BaseMetric):
    def __init__(self, provider="groq"):
        self.client = JudgeClient(provider=provider)

    def compute(self, query, response, contexts=None, reference=None) -> MetricResult:
        if not response:
            return MetricResult(name="Answer Relevancy", score=0.0)
        
        prompt = f"Query: {query}\nResponse: {response}\nRate how well the response addresses the query. Score 0-10. Output: SCORE: [val]"
        raw_output = self.client.ask_judge(prompt)
        
        score_match = re.search(r"SCORE:\s*([\d\.]+)", raw_output)
        score = float(score_match.group(1)) / 10.0 if score_match else 0.0 

        return MetricResult(name="Answer Relevancy", score=score)

# TODO: The parsing logic can extracted to a common utility function in ./utils
# TODO: The prompt templates can also be externalized for easier customization