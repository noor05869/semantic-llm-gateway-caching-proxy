"""
⚡ Model Router & Tokenomics Calculator (router.py)

Enterprise-grade model complexity router and token cost calculator:
1. Intent & Complexity Classifier (Deterministic heuristics + pattern matching).
2. Model Tiering: Tier 1 (Flash / Low-Cost) vs Tier 2 (Pro / Deep-Reasoning).
3. Token usage estimation & precise USD ($) cost accounting.
4. Cost savings delta calculator.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ModelTier(str, Enum):
    """Model performance and pricing tiers."""
    TIER_1_FAST = "tier_1_fast"        # Cheap, ultra-fast (Gemini 2.5 Flash)
    TIER_2_REASONING = "tier_2_pro"    # Heavy reasoning, architecture (Gemini 2.5 Pro)


@dataclass
class ModelPricing:
    """Pricing in USD per 1 Million tokens."""
    model_name: str
    tier: ModelTier
    input_cost_per_million: float
    output_cost_per_million: float

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate total USD cost for a given token usage."""
        input_cost = (prompt_tokens / 1_000_000.0) * self.input_cost_per_million
        output_cost = (completion_tokens / 1_000_000.0) * self.output_cost_per_million
        return round(input_cost + output_cost, 7)


# Standard pricing catalog (Google Gemini official production rates)
MODEL_CATALOG: Dict[str, ModelPricing] = {
    # Tier 1 Models
    "gemini-2.5-flash": ModelPricing(
        model_name="gemini-2.5-flash",
        tier=ModelTier.TIER_1_FAST,
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        model_name="gemini-2.5-flash-lite",
        tier=ModelTier.TIER_1_FAST,
        input_cost_per_million=0.0375,
        output_cost_per_million=0.15,
    ),
    # Tier 2 Models
    "gemini-2.5-pro": ModelPricing(
        model_name="gemini-2.5-pro",
        tier=ModelTier.TIER_2_REASONING,
        input_cost_per_million=1.25,
        output_cost_per_million=5.00,
    ),
}

DEFAULT_TIER_1_MODEL = "gemini-2.5-flash"
DEFAULT_TIER_2_MODEL = "gemini-2.5-pro"


@dataclass
class RoutingDecision:
    """The outcome of the complexity classification and routing process."""
    selected_model: str
    tier: ModelTier
    complexity_score: float  # 0.0 (Trivial) to 1.0 (Highly Complex)
    reasoning: List[str]
    is_override: bool = False


class ComplexityRouter:
    """
    Deterministic rule-based intent and complexity classifier.
    Runs in < 0.1ms without external API dependencies.
    """

    # Signals that indicate complex cognitive or technical tasks
    CODE_KEYWORDS = {
        "function", "class", "def ", "refactor", "algorithm", "regex",
        "sql", "database schema", "debug", "unit test", "typescript",
        "dockerfile", "kubernetes", "race condition", "deadlock", "recursion",
    }

    REASONING_KEYWORDS = {
        "architect", "step-by-step", "pros and cons", "trade-offs",
        "mathematical proof", "derive", "counterargument", "comprehensive plan",
        "detailed analysis", "system design", "benchmark", "root cause",
    }

    # Signals that indicate simple, low-complexity tasks
    SIMPLE_KEYWORDS = {
        "translate", "summarize in one sentence", "capital of", "who wrote",
        "correct grammar", "spelling", "synonym", "antonym", "define",
        "what is", "sentiment of", "yes or no", "format as json",
    }

    def __init__(
        self,
        tier_1_model: str = DEFAULT_TIER_1_MODEL,
        tier_2_model: str = DEFAULT_TIER_2_MODEL,
        complexity_threshold: float = 0.50,
    ):
        self.tier_1_model = tier_1_model
        self.tier_2_model = tier_2_model
        self.complexity_threshold = complexity_threshold

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Fast token estimation based on character and whitespace heuristics.
        Standard rule of thumb: ~4 characters per token in English.
        """
        if not text:
            return 0
        char_count = len(text)
        word_count = len(text.split())
        # Blended estimation: max of chars/4 and words * 1.3
        return max(1, int(round((char_count / 4.0 + word_count * 1.3) / 2.0)))

    def evaluate_complexity(self, prompt: str) -> Tuple[float, List[str]]:
        """
        Calculate a normalized complexity score between 0.0 and 1.0.
        Returns: (score, list_of_detected_reasons)
        """
        score = 0.0
        reasons = []
        lower = prompt.lower()
        word_count = len(prompt.split())

        # Factor 1: Code presence or request
        code_matches = [kw for kw in self.CODE_KEYWORDS if kw in lower]
        has_code_block = "```" in prompt or "    " in prompt
        if has_code_block or len(code_matches) >= 2:
            score += 0.50
            reasons.append(f"Contains code patterns/keywords: {code_matches[:3]}")
        elif len(code_matches) == 1:
            score += 0.25
            reasons.append(f"Single code keyword: '{code_matches[0]}'")

        # Factor 2: Deep reasoning / Architectural requests
        reasoning_matches = [kw for kw in self.REASONING_KEYWORDS if kw in lower]
        if len(reasoning_matches) >= 2:
            score += 0.55
            reasons.append(f"Multi-step reasoning keywords: {reasoning_matches[:3]}")
        elif len(reasoning_matches) == 1:
            score += 0.30
            reasons.append(f"Reasoning keyword: '{reasoning_matches[0]}'")

        # Factor 3: Prompt length / Context payload
        if word_count > 350:
            score += 0.35
            reasons.append(f"Long context prompt ({word_count} words)")
        elif word_count > 150:
            score += 0.15
            reasons.append(f"Moderate prompt length ({word_count} words)")

        # Factor 4: Simple task detection (Down-weights complexity)
        simple_matches = [kw for kw in self.SIMPLE_KEYWORDS if kw in lower]
        if simple_matches and word_count < 60:
            score -= 0.30
            reasons.append(f"Simple factual/utility task detected: '{simple_matches[0]}'")

        # Clamp between 0.0 and 1.0
        final_score = max(0.0, min(1.0, round(score, 2)))
        if not reasons:
            reasons.append("Default conversational request")

        return final_score, reasons

    def route(self, prompt: str, explicit_model: Optional[str] = None) -> RoutingDecision:
        """
        Route prompt to the most cost-effective model tier.
        Allows explicit client override if specified.
        """
        # Case A: Client explicitly asked for a specific model
        if explicit_model:
            tier = (
                ModelTier.TIER_2_REASONING
                if "pro" in explicit_model.lower()
                else ModelTier.TIER_1_FAST
            )
            return RoutingDecision(
                selected_model=explicit_model,
                tier=tier,
                complexity_score=1.0 if tier == ModelTier.TIER_2_REASONING else 0.0,
                reasoning=[f"Client explicit override requested: '{explicit_model}'"],
                is_override=True,
            )

        # Case B: Automatic complexity evaluation
        score, reasons = self.evaluate_complexity(prompt)

        if score >= self.complexity_threshold:
            return RoutingDecision(
                selected_model=self.tier_2_model,
                tier=ModelTier.TIER_2_REASONING,
                complexity_score=score,
                reasoning=reasons,
                is_override=False,
            )
        else:
            return RoutingDecision(
                selected_model=self.tier_1_model,
                tier=ModelTier.TIER_1_FAST,
                complexity_score=score,
                reasoning=reasons,
                is_override=False,
            )

    @staticmethod
    def calculate_savings(
        actual_model: str,
        prompt_tokens: int,
        completion_tokens: int,
        benchmark_model: str = DEFAULT_TIER_2_MODEL,
    ) -> Dict[str, float]:
        """
        Calculate token costs and dollar savings compared to blindly calling
        the expensive Tier 2 frontier model.
        """
        actual_pricing = MODEL_CATALOG.get(actual_model, MODEL_CATALOG[DEFAULT_TIER_1_MODEL])
        benchmark_pricing = MODEL_CATALOG.get(benchmark_model, MODEL_CATALOG[DEFAULT_TIER_2_MODEL])

        actual_cost = actual_pricing.calculate_cost(prompt_tokens, completion_tokens)
        benchmark_cost = benchmark_pricing.calculate_cost(prompt_tokens, completion_tokens)
        dollars_saved = max(0.0, benchmark_cost - actual_cost)
        pct_saved = (dollars_saved / benchmark_cost * 100.0) if benchmark_cost > 0 else 0.0

        return {
            "actual_cost_usd": actual_cost,
            "benchmark_cost_usd": benchmark_cost,
            "dollars_saved": round(dollars_saved, 7),
            "percentage_saved": round(pct_saved, 2),
        }


# =====================================================================
# Standalone Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Complexity Router & Cost Calculator: Verification Suite")
    print("=" * 65)

    router = ComplexityRouter(complexity_threshold=0.50)

    test_queries = [
        # (Prompt, Expected Tier)
        (
            "What is the capital of France?",
            ModelTier.TIER_1_FAST,
        ),
        (
            "Translate 'Good morning, how are you today?' into Japanese.",
            ModelTier.TIER_1_FAST,
        ),
        (
            "Architect a distributed microservice for payment processing with idempotency and fault tolerance. Provide trade-offs.",
            ModelTier.TIER_2_REASONING,
        ),
        (
            "Here is my Python function:\ndef calculate_total(items):\n    return sum(items)\nCan you refactor it to handle edge cases and write unit tests?",
            ModelTier.TIER_2_REASONING,
        ),
    ]

    print("\n🔍 Testing Automatic Complexity Routing:\n")
    for prompt, expected_tier in test_queries:
        decision = router.route(prompt)
        status = "✅ PASS" if decision.tier == expected_tier else "❌ FAIL"

        print(f"Prompt: \"{prompt[:60]}...\"")
        print(f"-> Selected Model: [{decision.selected_model}] ({decision.tier.value})")
        print(f"-> Complexity Score: {decision.complexity_score:.2f} | Status: {status}")
        print(f"-> Detected Signals: {', '.join(decision.reasoning)}")
        print("-" * 65)

    # Token and Savings Test
    print("\n💰 Testing Tokenomics & Cost Savings Calculation:")
    sample_prompt_tokens = 500
    sample_completion_tokens = 300

    savings = router.calculate_savings(
        actual_model="gemini-2.5-flash",
        prompt_tokens=sample_prompt_tokens,
        completion_tokens=sample_completion_tokens,
        benchmark_model="gemini-2.5-pro",
    )

    print(f"Scenario: Simple query routed to Gemini 2.5 Flash instead of Pro")
    print(f"Prompt Tokens: {sample_prompt_tokens} | Completion Tokens: {sample_completion_tokens}")
    print(f"Actual Cost (Flash):      ${savings['actual_cost_usd']:.6f}")
    print(f"Benchmark Cost (Pro):     ${savings['benchmark_cost_usd']:.6f}")
    print(f"💵 Total Dollars Saved:   ${savings['dollars_saved']:.6f} ({savings['percentage_saved']}%)")

    print("\n" + "=" * 65)
    print("🎯 Step 3 Verification Complete: Intelligent Routing & Cost Engine Ready!")
    print("=" * 65)
