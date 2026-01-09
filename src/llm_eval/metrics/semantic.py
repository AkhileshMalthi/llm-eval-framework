from bert_score import BERTScorer
from llm_eval.metrics.base import BaseMetric, MetricResult

class BERTScoreMetric(BaseMetric):
    def __init__(self):
        # We use a lightweight model to keep it fast
        self.scorer = BERTScorer(model_type="distilbert-base-uncased", lang="en")

    def compute(self, query, response, contexts=None, reference=None) -> MetricResult:
        if not reference:
            return MetricResult(name="BERTScore", score=0.0)
            
        P, R, F1 = self.scorer.score([response], [reference])
        return MetricResult(name="BERTScore", score=float(F1.mean())) # type: ignore
