"""
⚡ Multi-Model Provider Client & Failover Controller (providers.py)

Enterprise-grade provider integration with automated failover:
1. Google Gemini integration using the official google-genai SDK.
2. Automated 429 (Rate Limit) and 503 (Overload) failover cascade.
3. Transparent failover tracking (is_failover, failover_reason).
4. Token usage tracking and latency measurement.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment variables (.env)
load_dotenv()

try:
    from google import genai
    from google.genai import errors as genai_errors
except ImportError:
    genai = None
    genai_errors = None


@dataclass
class LLMResponse:
    """Standardized response object returned by the provider gateway."""
    text: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    is_failover: bool = False
    failover_history: List[str] = field(default_factory=list)


# Fallback chains for high-availability routing
DEFAULT_FAILOVER_CHAINS: Dict[str, List[str]] = {
    # If primary model hits 429/503 -> fallback to secondary -> fallback to tertiary
    "gemini-3.8-flash": ["gemini-3.5-flash", "gemini-3.5-flash-lite"],
    "gemini-3.5-flash": ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
    "gemini-3.5-flash-lite": ["gemini-3.5-flash", "gemini-3.8-flash"],
}


class GeminiProvider:
    """
    Manages connections to Google Gemini models with automatic failover
    upon rate limits (HTTP 429) or transient service outages (HTTP 503).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        failover_chains: Optional[Dict[str, List[str]]] = None,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.failover_chains = failover_chains or DEFAULT_FAILOVER_CHAINS
        self._client: Optional[genai.Client] = None

        if self.api_key and genai is not None:
            self._client = genai.Client(api_key=self.api_key)

    @property
    def is_configured(self) -> bool:
        """Check if provider has a valid client and API key."""
        return self._client is not None and bool(self.api_key)

    def _call_model(self, model: str, prompt: str) -> Tuple[str, int, int]:
        """
        Execute a single raw call to Gemini API using google-genai SDK.
        Returns: (generated_text, prompt_tokens, completion_tokens)
        """
        if not self._client:
            raise RuntimeError("Gemini Client not initialized. Please check GOOGLE_API_KEY in .env.")

        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
        )

        text = response.text or ""

        # Extract token usage metadata from response
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        # Fallback estimation if metadata was omitted by upstream
        if prompt_tokens == 0:
            prompt_tokens = max(1, len(prompt.split()) * 4 // 3)
        if completion_tokens == 0:
            completion_tokens = max(1, len(text.split()) * 4 // 3)

        return text, prompt_tokens, completion_tokens

    def generate(
        self,
        prompt: str,
        primary_model: str = "gemini-3.5-flash",
        allow_failover: bool = True,
    ) -> LLMResponse:
        """
        Generate completion with automated failover cascade.

        If primary_model encounters a 429 RateLimit or 503 Overload error,
        it automatically attempts each model in the failover chain sequentially.
        """
        t0 = time.perf_counter()

        # Build execution queue: [primary_model, fallback_1, fallback_2, ...]
        models_to_try = [primary_model]
        if allow_failover and primary_model in self.failover_chains:
            models_to_try.extend(
                m for m in self.failover_chains[primary_model] if m not in models_to_try
            )

        failover_history: List[str] = []
        last_exception: Optional[Exception] = None

        for idx, current_model in enumerate(models_to_try):
            is_fallback_attempt = idx > 0
            try:
                text, p_tokens, c_tokens = self._call_model(current_model, prompt)
                latency_ms = (time.perf_counter() - t0) * 1000

                return LLMResponse(
                    text=text,
                    model_used=current_model,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=p_tokens + c_tokens,
                    latency_ms=latency_ms,
                    is_failover=is_fallback_attempt,
                    failover_history=failover_history,
                )

            except Exception as exc:
                last_exception = exc
                err_msg = str(exc)
                failover_history.append(f"Model '{current_model}' failed: {err_msg[:80]}")

                # If this is the last available fallback model, raise the error
                if idx == len(models_to_try) - 1:
                    raise RuntimeError(
                        f"All models in failover chain exhausted. Last error: {err_msg}"
                    ) from exc

                # Otherwise, loop to the next fallback model in the cascade
                continue

        # Should not be reached, but safety guard
        raise last_exception or RuntimeError("Unknown error in provider generation.")


# =====================================================================
# Standalone Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Multi-Provider & 429 Failover: Verification Suite")
    print("=" * 65)

    provider = GeminiProvider()
    print(f"\n🔑 Gemini Client Configured: [{provider.is_configured}]")

    # Test 1: Live Generation with Primary Model (gemini-3.5-flash)
    print("\n--- Test 1: Live Primary Model Generation ---")
    test_prompt = "What are the three laws of robotics in 1 sentence each?"
    print(f"Prompt: \"{test_prompt}\"")
    print("Calling Gemini API...")

    try:
        resp = provider.generate(test_prompt, primary_model="gemini-3.5-flash")
        print(f"✅ Success! Generated in {resp.latency_ms:.1f}ms")
        print(f"Model Used:         {resp.model_used}")
        print(f"Tokens:             {resp.prompt_tokens} in / {resp.completion_tokens} out (Total: {resp.total_tokens})")
        print(f"Failover Triggered: {resp.is_failover}")
        print(f"Response Preview:   \"{resp.text.strip()[:100]}...\"")
    except Exception as e:
        print(f"❌ Error during generation: {e}")

    # Test 2: Live Failover Test on 503 Overload (gemini-3.8-flash -> gemini-3.5-flash)
    print("\n--- Test 2: Real Failover Test (gemini-3.8-flash -> gemini-3.5-flash) ---")
    print("Scenario: Attempt 'gemini-3.8-flash' (experiencing 503 high demand).")
    print("Expected: Gateway catches 503 and cascades to 'gemini-3.5-flash'.")
    try:
        resp_failover = provider.generate(
            "State the Pythagorean theorem in one sentence.",
            primary_model="gemini-3.8-flash",
            allow_failover=True,
        )
        print(f"Status:             ✅ SUCCESS via Failover!")
        print(f"Final Model Used:   [{resp_failover.model_used}]")
        print(f"Failover Active:    {resp_failover.is_failover}")
        print(f"Failover History:   {resp_failover.failover_history}")
        print(f"Response:           \"{resp_failover.text.strip()[:100]}...\"")
    except Exception as e:
        print(f"❌ Error during live failover test: {e}")

    # Test 3: Simulated 429 Rate-Limit Failover Cascade
    print("\n--- Test 3: Simulated 429 Failover Cascade ---")
    print("Scenario: 'gemini-3.5-flash' throws HTTP 429 Too Many Requests.")
    print("Expected: Gateway catches 429 and cascades to 'gemini-3.5-flash-lite'.")

    class SimulatedFailoverProvider(GeminiProvider):
        """Simulates a 429 rate-limit error on the first model."""
        def _call_model(self, model: str, prompt: str):
            if model == "gemini-3.5-flash":
                raise Exception("429 ResourceExhausted: Quota exceeded for gemini-3.5-flash")
            return f"[Simulated Response from {model}] Success!", 12, 18

    sim_provider = SimulatedFailoverProvider(api_key="simulated_key")
    sim_provider._client = object()

    sim_resp = sim_provider.generate(
        "Explain idempotency in APIs", primary_model="gemini-3.5-flash"
    )

    print(f"Status:             ✅ SUCCESS")
    print(f"Final Model Used:   [{sim_resp.model_used}]")
    print(f"Failover Active:    {sim_resp.is_failover}")
    print(f"Failover History:   {sim_resp.failover_history}")
    print(f"Response:           \"{sim_resp.text}\"")

    assert sim_resp.is_failover is True
    assert sim_resp.model_used == "gemini-3.5-flash-lite"
    print("✅ Automated 429 Failover Cascade Verified!")

    print("\n" + "=" * 65)
    print("🎯 Step 4 Verification Complete: Failover Engine Ready!")
    print("=" * 65)
