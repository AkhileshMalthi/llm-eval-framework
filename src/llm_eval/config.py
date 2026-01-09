from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional

class LLMJudgeConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    dimensions: List[str] = ["coherence", "relevance", "safety"]

class EvalConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    eval_name: str
    dataset_path: str
    output_dir: str = "results"
    metrics: List[str]
    llm_judge: Optional[LLMJudgeConfig] = None
    
    @field_validator('metrics')
    @classmethod
    def validate_metrics_not_empty(cls, v):
        if not v:
            raise ValueError("Metrics list cannot be empty")
        return v