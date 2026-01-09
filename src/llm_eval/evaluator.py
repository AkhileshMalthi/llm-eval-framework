from pathlib import Path
import pandas as pd
from typing import List, Sequence
from llm_eval.metrics.base import BaseMetric

class Evaluator:
    def __init__(self, metrics: Sequence[BaseMetric]):
        """Initialize evaluator with metrics.
        
        Args:
            metrics: Sequence of metric instances to use for evaluation.
        """
        self.metrics = metrics

    def run(self, dataset_path: str | Path):
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        df = pd.read_json(dataset_path, lines=True)
        results = []

        for _, row in df.iterrows():
            row_results = {"query": row['query'], "response": row['response']}
            for metric in self.metrics:
                res = metric.compute(
                    query=row['query'],
                    response=row.get('response', ""),
                    contexts=row.get('retrieved_contexts', []),
                    reference=row.get('expected_answer', "")
                )
                row_results[res.name] = res.score
            results.append(row_results)
        
        return pd.DataFrame(results)