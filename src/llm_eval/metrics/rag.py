import re
from llm_eval.metrics.base import BaseMetric, MetricResult
from llm_eval.utils.llm_client import JudgeClient

class FaithfulnessMetric(BaseMetric):
    def __init__(self):
        self.client = JudgeClient()

    def compute(self, query, response, contexts=None, reference=None) -> MetricResult:
        context_str = "\n".join(contexts if contexts is not None else [])
        prompt = f"""
        You are an auditor. Verify if the following response is 100% supported by the context.
        
        Context: {context_str if context_str else 'N/A'}
        Response: {response}
        
        Rules:
        1. If the response contains info NOT in the context, score is 0.
        2. If the response is fully supported, score is 1.
        
        Output format:
        SCORE: [0 or 1]
        REASON: [Short explanation]
        """
        
        raw_output = self.client.ask_judge(prompt)
        
        # Simple parsing logic
        score_match = re.search(r"SCORE:\s*([\d\.]+)", raw_output)
        score = float(score_match.group(1)) if score_match else 0.0
        
        return MetricResult(name="Faithfulness", score=score, reasoning=raw_output)