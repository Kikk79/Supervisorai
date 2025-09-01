from dataclasses import dataclass, field
from typing import List, Dict, Any
import time

@dataclass
class LLMCallRecord:
    """Represents a single call to the LLM."""
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

class CostTracker:
    """
    Tracks the cost of LLM calls based on token usage and model pricing.
    """
    # Pricing per million tokens (as of a hypothetical date)
    MODEL_PRICING = {
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "default": {"input": 1.00, "output": 5.00} # A default for unknown models
    }

    def __init__(self):
        self.call_history: List[LLMCallRecord] = []

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculates the cost of a single LLM call."""
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["default"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def log_call(self, model: str, input_tokens: int, output_tokens: int, context: Dict[str, Any] = None):
        """Logs a new LLM call and its associated cost."""
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        record = LLMCallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            context=context or {}
        )
        self.call_history.append(record)
        print(f"Logged LLM call. Model: {model}, Cost: ${cost:.6f}")

    def get_total_cost(self) -> float:
        """Returns the total accumulated cost of all LLM calls."""
        return sum(record.cost for record in self.call_history)

    def get_cost_report(self) -> Dict[str, Any]:
        """Generates a summary report of all costs."""
        total_cost = self.get_total_cost()
        total_calls = len(self.call_history)

        cost_by_model = {}
        for record in self.call_history:
            cost_by_model[record.model] = cost_by_model.get(record.model, 0) + record.cost

        return {
            "total_cost": total_cost,
            "total_calls": total_calls,
            "cost_by_model": cost_by_model,
            "average_cost_per_call": total_cost / total_calls if total_calls > 0 else 0
        }
