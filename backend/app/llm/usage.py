"""Token usage + cost accounting.

Prices are USD per 1M tokens (input, output) as published by OpenAI (checked 18 Nov 2026). They are
estimates for the UI; override with PRICING_JSON in .env if your account differs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..config import settings

PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-6-astra": (10.0, 50.0),
    "gpt-6.1-sol": (2.0, 10.0),
    "gpt-6-sol": (2.0, 10.0),
    "gpt-6.1-terra": (1.0, 5.0),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.6-sol": (4.0, 20.0),
    "gpt-5.6": (4.0, 20.0),
    "gpt-5.5": (5.0, 30.0),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-3-small": (0.02, 0.0),
}
if settings.pricing_json:
    try:
        PRICES.update({k: tuple(v) for k, v in json.loads(settings.pricing_json).items()})
    except Exception:  # pragma: no cover
        pass


def price(model: str) -> tuple[float, float] | None:
    if model in PRICES:
        return PRICES[model]
    # dated snapshots e.g. gpt-4o-mini-2024-07-18 (longest prefix wins)
    for k in sorted(PRICES, key=len, reverse=True):
        if model.startswith(k):
            return PRICES[k]
    return None


@dataclass
class Usage:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, model: str, input_tokens: int, output_tokens: int, ms: float | None = None) -> None:
        p = price(model)
        cost = (input_tokens * p[0] + output_tokens * p[1]) / 1e6 if p else 0.0
        self.steps.append({"step": step, "model": model, "input_tokens": int(input_tokens),
                           "output_tokens": int(output_tokens), "cost_usd": cost, "priced": p is not None,
                           **({"ms": round(ms, 1)} if ms is not None else {})})

    def extend(self, other: "Usage") -> None:
        self.steps.extend(other.steps)

    @property
    def cost(self) -> float:
        return sum(s["cost_usd"] for s in self.steps)

    @property
    def input_tokens(self) -> int:
        return sum(s["input_tokens"] for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s["output_tokens"] for s in self.steps)

    def summary(self) -> dict:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost, 6), "steps": self.steps}
