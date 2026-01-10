from typing import Optional
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from llm_eval.metrics.base import BaseMetric, MetricResult

class BleuMetric(BaseMetric):
    def compute(self, query:str, response:str, contexts: Optional[list[str]] = None, reference: Optional[str] = None) -> MetricResult:
        if reference is None:
            raise ValueError("Reference text is required for BLEU metric.")
        
        reference_tokens = [nltk.word_tokenize(reference.lower())]
        candidate_tokens = nltk.word_tokenize(response.lower())
        
        # Use smoothing to avoid warnings for zero n-gram matches
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)

        if not isinstance(score, float):
            raise ValueError("BLEU score computation did not return a float value.")
        
        return MetricResult(name="BLEU", score=score)

class RougeLMetric(BaseMetric):
    def compute(self, query, response, contexts=None, reference=None) -> MetricResult:
        if reference is None:
            raise ValueError("Reference text is required for ROUGE-L metric.")
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, response)
        return MetricResult(name="ROUGE-L", score=scores['rougeL'].fmeasure)

# Note: Classical metrics (BLEU, ROUGE) don't use 'query' and 'contexts' parameters,
# but they are required by the BaseMetric interface to maintain a uniform signature
# across all metrics. This enables polymorphic usage and consistent API design.