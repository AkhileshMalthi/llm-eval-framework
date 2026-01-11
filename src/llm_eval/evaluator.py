from pathlib import Path
import pandas as pd
from typing import List, Sequence
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_eval.metrics.base import BaseMetric

class Evaluator:
    def __init__(self, metrics: Sequence[BaseMetric], max_workers: int = 4):
        """Initialize evaluator with metrics.
        
        Args:
            metrics: Sequence of metric instances to use for evaluation.
            max_workers: Maximum number of parallel workers for evaluation. Default is 4.
        """
        self.metrics = metrics
        self.max_workers = max_workers

    def _process_row(self, row_data):
        """Process a single row with all metrics.
        
        This method is designed to be called in parallel for different rows.
        
        Args:
            row_data: Tuple of (index, row) from DataFrame.iterrows()
            
        Returns:
            Dictionary containing query, response, and all metric scores.
        """
        _, row = row_data
        row_results = {"query": row['query'], "response": row['response']}
        
        for metric in self.metrics:
            res = metric.compute(
                query=row['query'],
                response=row.get('response', ""),
                contexts=row.get('retrieved_contexts', []),
                reference=row.get('expected_answer', "")
            )
            row_results[res.name] = res.score
        
        return row_results

    def run(self, dataset_path: str | Path):
        """Run evaluation on the dataset with parallel processing.
        
        Args:
            dataset_path: Path to the JSONL dataset file.
            
        Returns:
            DataFrame containing evaluation results for all rows.
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        df = pd.read_json(dataset_path, lines=True)
        
        # Convert to list of (index, row) tuples for parallel processing
        rows = list(df.iterrows())
        
        # Use ThreadPoolExecutor for parallel processing with tqdm progress bar
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks and track them
            futures = {executor.submit(self._process_row, row): i for i, row in enumerate(rows)}
            
            # Collect results as they complete, with progress bar
            results: list[dict | None] = [None] * len(rows) 
            for future in tqdm(as_completed(futures), total=len(rows), desc="Processing items"):
                idx = futures[future]
                results[idx] = future.result()
        
        return pd.DataFrame(results)