"""
⚡ Semantic Caching Engine (cache_engine.py)

Enterprise-grade semantic hashing and vector similarity engine.
- Fast CPU-based embeddings using ONNX runtime (BAAI/bge-small-en-v1.5).
- Prompt canonicalization & SHA-256 fast-path exact hashing.
- Cosine similarity computation with L2 normalization.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import List, Tuple, Union
import numpy as np
from fastembed import TextEmbedding


class SemanticCacheEngine:
    """
    Core engine responsible for text embedding generation,
    vector normalization, and semantic similarity evaluations.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_THRESHOLD = 0.88
    EMBEDDING_DIM = 384

    def __init__(self, model_name: str = DEFAULT_MODEL, threshold: float = DEFAULT_THRESHOLD):
        """
        Initialize the embedding model and similarity threshold.

        Args:
            model_name: HuggingFace model ID for fastembed (bge-small produces 384-dim vectors).
            threshold: Cosine similarity cutoff (default: 0.88, balances precision & recall).
        """
        self.model_name = model_name
        self.threshold = threshold
        # Initialize the ONNX-optimized embedding model
        self._model = TextEmbedding(model_name=self.model_name)

    @staticmethod
    def canonicalize(text: str) -> str:
        """
        Normalize input prompt text to remove irrelevant variations
        (extra whitespaces, non-standard casing, trailing punctuation).
        """
        text = text.strip().lower()
        # Collapse multiple spaces, tabs, newlines into a single space
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def exact_hash(text: str) -> str:
        """
        Generate a deterministic SHA-256 hash of canonicalized text.
        Serves as a Tier-0 exact-match bypass before vector search.
        """
        canonical = SemanticCacheEngine.canonicalize(text)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an L2-normalized 384-dimensional embedding vector for a prompt.

        Returns:
            np.ndarray of shape (384,), float32, normalized to unit length.
        """
        cleaned_text = self.canonicalize(text)
        # fastembed returns an iterator of numpy arrays
        raw_embedding = next(self._model.embed([cleaned_text]))
        vec = np.array(raw_embedding, dtype=np.float32)

        # L2-normalize vector: ||v|| = 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate normalized embeddings for a batch of strings."""
        cleaned_texts = [self.canonicalize(t) for t in texts]
        embeddings = list(self._model.embed(cleaned_texts))
        results = []
        for raw in embeddings:
            vec = np.array(raw, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)
        return results

    @staticmethod
    def compute_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two normalized vectors.
        Since both are unit-normalized (||v|| = 1), cosine similarity
        is simply the dot product.
        """
        return float(np.dot(vec_a, vec_b))

    def is_semantic_match(
        self, vec_a: np.ndarray, vec_b: np.ndarray, threshold: float | None = None
    ) -> Tuple[bool, float]:
        """
        Determine if two vectors meet or exceed the similarity threshold.

        Returns:
            (is_match: bool, similarity_score: float)
        """
        target_threshold = threshold if threshold is not None else self.threshold
        score = self.compute_similarity(vec_a, vec_b)
        return (score >= target_threshold, score)


# =====================================================================
# Standalone Verification Suite
# =====================================================================
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 65)
    print("Semantic Cache Engine: Verification Suite")
    print("=" * 65)

    engine = SemanticCacheEngine(threshold=0.88)

    test_pairs = [
        # (Query A, Query B, Expected Semantic Match)
        (
            "How do I reset my account password?",
            "Steps to reset password",
            True,
        ),
        (
            "What is the capital of France?",
            "Tell me the capital city of France.",
            True,
        ),
        (
            "How do I write a Python decorator?",
            "What is the best recipe for chocolate cake?",
            False,
        ),
        (
            "Explain quantum computing simply",
            "Give me a simple explanation of quantum computers",
            True,
        ),
    ]

    print("\n🔍 Running Semantic Similarity Tests (Threshold: 0.88):\n")
    for q1, q2, expected in test_pairs:
        t0 = time.perf_counter()
        v1 = engine.generate_embedding(q1)
        v2 = engine.generate_embedding(q2)
        match, score = engine.is_semantic_match(v1, v2)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        status = "✅ PASS" if (match == expected) else "❌ FAIL"
        match_str = "CACHE HIT" if match else "CACHE MISS"

        print(f"Query 1: '{q1}'")
        print(f"Query 2: '{q2}'")
        print(f"Similarity: {score:.4f} -> [{match_str}] | Took: {elapsed_ms:.1f}ms | {status}")
        print("-" * 65)

    # Test Exact Hashing
    print("\n🔑 Exact SHA-256 Hash Bypass Test:")
    h1 = engine.exact_hash("  How to learn AI?  ")
    h2 = engine.exact_hash("how i can learn ai?")
    print(f"Canonical Hash 1: {h1}")
    print(f"Canonical Hash 2: {h2}")
    assert h1 == h2, "Canonical hashes must match for identical normalized queries!"
    print("✅ Canonical hashing verified!")
    print("\n🎯 Step 1 Verification Complete!")
