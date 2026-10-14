"""
Token and cost accounting module for FreightTiger Shipping Cost Assistant.
Tracks LLM calls, prompt/completion tokens, and calculates costs based on published pricing.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LLMCallRecord:
    route: str
    week_of: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    success: bool
    fallback_used: bool


class TokenCostLogger:
    """
    Records and calculates token usage and dollar costs.
    Pricing per 1M tokens (USD):
    - groq:llama-3.3-70b-versatile: $0.59 / 1M prompt, $0.79 / 1M completion
    - gpt-4o-mini: $0.15 / 1M prompt, $0.60 / 1M completion
    - deterministic: $0.00
    """

    PRICING_PER_1M = {
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "deterministic": {"input": 0.0, "output": 0.0},
    }

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self.records: List[LLMCallRecord] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.total_cost_usd = 0.0

    def record_call(
        self,
        route: str,
        week_of: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool = True,
        fallback_used: bool = False,
    ) -> float:
        total_tokens = prompt_tokens + completion_tokens
        rates = self.PRICING_PER_1M.get(model, {"input": 0.59, "output": 0.79})
        cost = (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000.0

        record = LLMCallRecord(
            route=route,
            week_of=week_of,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            success=success,
            fallback_used=fallback_used,
        )

        self.records.append(record)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1
        self.total_cost_usd += cost

        return cost

    def get_summary(self) -> Dict[str, any]:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "pricing_reference": self.PRICING_PER_1M,
        }

    def save_log(self, filepath: Optional[Path] = None):
        target = filepath or self.log_path
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()
        summary["detailed_records"] = [asdict(r) for r in self.records]
        with open(target, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
