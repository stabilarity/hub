from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum

class UIBDimension(str, Enum):
    CAUSAL = "causal"
    EMBODIED = "embodied"
    MULTIMODAL = "multimodal"
    TEMPORAL = "temporal"
    SOCIAL = "social"
    TOOL_CREATION = "tool_creation"
    TRANSFER = "transfer"
    EFFICIENCY = "efficiency"

class UIBRunRequest(BaseModel):
    api_key: str = Field(..., description="OpenRouter API key (or any OpenAI-compatible key)")
    api_base: str = Field(default="https://openrouter.ai/api/v1", description="API base URL")
    model: str = Field(..., description="Model identifier (e.g. openai/gpt-4o, anthropic/claude-3.5-sonnet)")
    dimensions: List[UIBDimension] = Field(default=list(UIBDimension), description="Dimensions to evaluate (default: all 8)")
    quick_mode: bool = Field(default=True, description="Quick mode (5 tasks/dim) vs full (25 tasks/dim)")
    benchmark_type: str = Field(default="uib", description="Benchmark type: uib, mmlu_mini, reasoning")

class DimensionScore(BaseModel):
    dimension: str
    score: float = Field(..., ge=0, le=100, description="Score 0-100")
    tasks_completed: int
    tasks_total: int
    avg_latency_ms: float
    cost_usd: float
    details: Optional[Dict] = None

class UIBResult(BaseModel):
    model: str
    benchmark_type: str = "uib"
    composite_score: float = Field(..., description="Weighted composite (0-100)")
    efficiency_normalized: float = Field(..., description="Score / cost normalization")
    dimensions: List[DimensionScore]
    total_cost_usd: float
    total_latency_ms: float
    timestamp: str
    run_id: str
    quick_mode: bool

class UIBLeaderboardEntry(BaseModel):
    rank: int
    model: str
    benchmark_type: str = "uib"
    avg_score: float
    min_score: float
    max_score: float
    avg_cost: float
    min_cost: float
    max_cost: float
    runs: int
    last_run: str

class UIBResultSummary(BaseModel):
    run_id: str
    model: str
    benchmark_type: str
    composite_score: float
    efficiency_normalized: float
    total_cost_usd: float
    timestamp: str
    dimensions: List[DimensionScore]

class UIBStatusResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.2.0"
    total_runs: int = 0
    models_evaluated: int = 0
    dimensions: List[str] = list(UIBDimension)
    supported_providers: List[str] = ["OpenRouter", "OpenAI", "Anthropic", "Google", "Together", "Local (Ollama/LMStudio)"]
    supported_benchmarks: List[str] = ["uib", "mmlu_mini", "reasoning"]
