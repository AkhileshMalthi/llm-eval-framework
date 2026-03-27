# Add a Custom Metric

This framework is designed to be extended. Adding a new metric requires implementing one abstract method and registering it with the CLI.

---

## Step 1: Create the metric class

All metrics live in `src/llm_eval/metrics/`. Create a new file or add to an existing module.

Your class must inherit from `BaseMetric` and implement `compute()`:

```python title="src/llm_eval/metrics/my_metric.py"
from typing import Optional
from llm_eval.metrics.base import BaseMetric, MetricResult


class MyMetric(BaseMetric):
    """
    A minimal custom metric example.
    
    Replace the body of compute() with your actual logic.
    """

    def compute(
        self,
        query: str,
        response: str,
        contexts: Optional[list[str]] = None,
        reference: Optional[str] = None,
    ) -> MetricResult:
        # Your scoring logic here.
        # score MUST be in [0, 1].
        score = 0.5  # placeholder

        return MetricResult(
            name="MyMetric",
            score=score,
            reasoning="Optional explanation string",  # set to None if not applicable
        )
```

### Rules

| Rule | Reason |
|------|--------|
| `score` must be in `[0, 1]` | Radar charts and summary statistics assume this range |
| `name` must be a unique string | Used as the column header in the results DataFrame |
| Never raise on missing `contexts` or `reference` | Return `score=0.0` with a `reasoning` explaining the miss |
| Use `Optional` types for `contexts` and `reference` | Not all datasets have both fields |

---

## Step 2: Register the metric in the CLI

Open `src/llm_eval/cli.py` and add your metric to the `registry` dict inside `get_metric_instances()`:

```python title="src/llm_eval/cli.py (partial)"
from llm_eval.metrics.my_metric import MyMetric  # add this import

def get_metric_instances(config):
    ...
    registry = {
        "bleu": BleuMetric(),
        "rouge": RougeLMetric(),
        ...
        "my_metric": MyMetric(),   # add this line
    }
    return [registry[m] for m in config.metrics if m in registry]
```

---

## Step 3: Use it in a config file

```yaml title="config.yaml"
eval_name: "Test my metric"
dataset_path: "benchmarks/rag_benchmark.jsonl"
metrics:
  - bleu
  - my_metric
```

---

## Step 4 (optional): Use it directly in Python

```python
from llm_eval import Evaluator
from llm_eval.metrics.my_metric import MyMetric

evaluator = Evaluator(metrics=[MyMetric()], max_workers=1)
results_df = evaluator.run("benchmarks/rag_benchmark.jsonl")
print(results_df["MyMetric"].describe())
```

---

## Example: exact-match metric

Here is a concrete example of a simple exact-match metric:

```python title="src/llm_eval/metrics/exact_match.py"
from typing import Optional
from llm_eval.metrics.base import BaseMetric, MetricResult


class ExactMatchMetric(BaseMetric):
    """Returns 1.0 if the response exactly matches the reference (case-insensitive)."""

    def compute(
        self,
        query: str,
        response: str,
        contexts: Optional[list[str]] = None,
        reference: Optional[str] = None,
    ) -> MetricResult:
        if reference is None:
            return MetricResult(
                name="ExactMatch",
                score=0.0,
                reasoning="No reference provided",
            )
        score = 1.0 if response.strip().lower() == reference.strip().lower() else 0.0
        return MetricResult(name="ExactMatch", score=score)
```

Register it as `"exact_match"` in the CLI registry, then add it to your config:

```yaml
metrics:
  - exact_match
```

---

## Example: LLM-based metric

If your metric needs to call an LLM, use `JudgeClient`:

```python title="src/llm_eval/metrics/conciseness.py"
from typing import Optional
from llm_eval.metrics.base import BaseMetric, MetricResult
from llm_eval.utils.llm_client import JudgeClient
from llm_eval.utils.parsing import extract_score_from_response


class ConcisenessMetric(BaseMetric):
    """Rates how concisely the response answers the query (0-10, LLM-graded)."""

    def __init__(self, provider: str = "openai"):
        self.client = JudgeClient(provider=provider)

    def compute(
        self,
        query: str,
        response: str,
        contexts: Optional[list[str]] = None,
        reference: Optional[str] = None,
    ) -> MetricResult:
        prompt = (
            f"Query: {query}\nResponse: {response}\n"
            "Rate how concisely this response answers the query. Score 0-10. Output: SCORE: [val]"
        )
        raw = self.client.ask_judge(prompt)
        score = extract_score_from_response(raw, max_score=10.0)
        return MetricResult(name="Conciseness", score=score, reasoning=raw)
```

---

## Tips

- **Keep `compute()` thread-safe** — the `Evaluator` calls it from multiple threads.
- **Avoid mutable state** — do not store per-call state on `self`.
- **Handle missing fields gracefully** — return `score=0.0` rather than raising.
- **Test your metric** — add a test file in `tests/` following the pattern of `tests/test_classical_metrics.py`.
