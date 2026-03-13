"""UIB Python Client"""
import requests
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class DimensionResult:
    dimension: str
    score: float
    tasks_completed: int
    tasks_total: int
    avg_latency_ms: float
    cost_usd: float

@dataclass
class UIBResult:
    model: str
    composite_score: float
    efficiency_normalized: float
    dimensions: List[DimensionResult]
    total_cost_usd: float
    total_latency_ms: float
    run_id: str

class UIBClient:
    def __init__(self, stabilarity_key: str, openrouter_key: str,
                 api_base: str = "https://openrouter.ai/api/v1",
                 gateway: str = "https://hub.stabilarity.com/api-gateway"):
        self.stabilarity_key = stabilarity_key
        self.openrouter_key = openrouter_key
        self.api_base = api_base
        self.gateway = gateway.rstrip("/")
    
    def run(self, model: str, dimensions: Optional[List[str]] = None,
            quick: bool = True) -> UIBResult:
        resp = requests.post(
            f"{self.gateway}/v1/uib/run",
            headers={"X-API-Key": self.stabilarity_key, "Content-Type": "application/json"},
            json={
                "api_key": self.openrouter_key,
                "api_base": self.api_base,
                "model": model,
                "dimensions": dimensions or ["causal","embodied","multimodal","temporal","social","tool_creation","transfer","efficiency"],
                "quick_mode": quick
            },
            timeout=300
        )
        resp.raise_for_status()
        data = resp.json()
        return UIBResult(
            model=data["model"],
            composite_score=data["composite_score"],
            efficiency_normalized=data["efficiency_normalized"],
            dimensions=[DimensionResult(**d) for d in data["dimensions"]],
            total_cost_usd=data["total_cost_usd"],
            total_latency_ms=data["total_latency_ms"],
            run_id=data["run_id"]
        )
    
    def leaderboard(self, limit: int = 20) -> List[Dict]:
        resp = requests.get(
            f"{self.gateway}/v1/uib/leaderboard",
            headers={"X-API-Key": self.stabilarity_key},
            params={"limit": limit}
        )
        resp.raise_for_status()
        return resp.json()
    
    def status(self) -> Dict:
        resp = requests.get(
            f"{self.gateway}/v1/uib/status",
            headers={"X-API-Key": self.stabilarity_key}
        )
        resp.raise_for_status()
        return resp.json()
