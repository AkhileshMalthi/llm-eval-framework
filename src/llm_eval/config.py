from pydantic import BaseModel, Field, ConfigDict
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