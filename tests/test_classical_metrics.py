from llm_eval.metrics.classical import BleuMetric, RougeLMetric
from llm_eval.metrics.base import MetricResult

def test_bleu_metric():
    bleu_metric = BleuMetric()
    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    contexts = []
    reference = "Paris is the capital of France."
    
    result: MetricResult = bleu_metric.compute(query=query, response=response, contexts=contexts, reference=reference)

    assert result.name == "BLEU"
    assert 0.0 <= result.score <= 1.0

def test_rouge_l_metric():
    rouge_l_metric = RougeLMetric()
    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    contexts = []
    reference = "Paris is the capital of France."
    
    result: MetricResult = rouge_l_metric.compute(query=query, response=response, contexts=contexts, reference=reference)
    
    assert result.name == "ROUGE-L"
    assert 0.0 <= result.score <= 1.0