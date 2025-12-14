"""
Evaluation metrics and benchmarking for RAG system

This module provides comprehensive evaluation capabilities for measuring:
- Retrieval quality (precision@k, recall@k, MRR)
- Answer quality (faithfulness, relevance, hallucination detection)
- Confidence calibration (do 70% confidence answers succeed 70% of the time?)
- System performance comparisons (with/without reranking, different thresholds)

Interview Keywords: evaluation metrics, precision/recall, calibration, A/B testing
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Comprehensive evaluator for RAG system performance.

    Interview talking points:
    - "I measure both retrieval and generation quality"
    - "I track confidence calibration to ensure trust"
    - "I use standard IR metrics like precision@k and MRR"
    """

    def __init__(self):
        """Initialize evaluator with tracking dictionaries."""
        self.metrics_history = []
        self.confidence_buckets = defaultdict(list)  # Track confidence vs actual performance

        logger.info("RAGEvaluator initialized")

    # ========================================================================
    # RETRIEVAL METRICS
    # ========================================================================

    def precision_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@k: What % of top-k retrieved docs are relevant?

        Formula: (# relevant docs in top-k) / k

        Example:
            Retrieved top-5: [doc1, doc2, doc3, doc4, doc5]
            Relevant: [doc1, doc3, doc5]
            Precision@5 = 3/5 = 0.6

        Interview point: "I use precision@k to measure retrieval accuracy"

        Args:
            retrieved_docs: List of document IDs in retrieval order
            relevant_docs: List of ground-truth relevant document IDs
            k: Number of top documents to consider

        Returns:
            Precision@k score (0.0 to 1.0)
        """
        if not retrieved_docs or k == 0:
            return 0.0

        top_k = retrieved_docs[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_docs))

        precision = relevant_in_top_k / k
        logger.debug(f"Precision@{k}: {precision:.3f}")

        return precision

    def recall_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@k: What % of all relevant docs are in top-k?

        Formula: (# relevant docs in top-k) / (total # relevant docs)

        Example:
            Retrieved top-5: [doc1, doc2, doc3, doc4, doc5]
            All relevant: [doc1, doc3, doc6, doc7]  # 4 total
            Recall@5 = 2/4 = 0.5 (found doc1 and doc3, missed doc6 and doc7)

        Interview point: "I measure recall to ensure we don't miss important information"

        Args:
            retrieved_docs: List of document IDs in retrieval order
            relevant_docs: List of ground-truth relevant document IDs
            k: Number of top documents to consider

        Returns:
            Recall@k score (0.0 to 1.0)
        """
        if not relevant_docs:
            return 0.0

        top_k = retrieved_docs[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_docs))

        recall = relevant_in_top_k / len(relevant_docs)
        logger.debug(f"Recall@{k}: {recall:.3f}")

        return recall

    def mean_reciprocal_rank(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR): Position of first relevant doc.

        Formula: 1 / (rank of first relevant doc)

        Example:
            Retrieved: [doc1, doc2, doc3, doc4, doc5]
            Relevant: [doc3, doc6]
            First relevant is doc3 at position 3
            MRR = 1/3 = 0.333

        Interview point: "MRR measures how quickly users find relevant info"

        Args:
            retrieved_docs: List of document IDs in retrieval order
            relevant_docs: List of ground-truth relevant document IDs

        Returns:
            MRR score (0.0 to 1.0, higher is better)
        """
        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                mrr = 1.0 / rank
                logger.debug(f"MRR: {mrr:.3f} (first relevant at rank {rank})")
                return mrr

        logger.debug("MRR: 0.0 (no relevant docs found)")
        return 0.0

    # ========================================================================
    # ANSWER QUALITY METRICS
    # ========================================================================

    def answer_faithfulness(
        self,
        answer: str,
        source_docs: List[str]
    ) -> float:
        """
        Measure if answer content comes from source documents (anti-hallucination).

        Simple version: Check if answer phrases appear in sources
        Production version: Use NLI model or LLM-as-judge

        Interview point: "I measure faithfulness to detect hallucinations"

        Args:
            answer: Generated answer text
            source_docs: List of source document texts

        Returns:
            Faithfulness score (0.0 to 1.0)
        """
        if not answer or not source_docs:
            return 0.0

        # Combine all source content
        source_text = " ".join(source_docs).lower()

        # Split answer into phrases (simple tokenization)
        answer_phrases = [p.strip() for p in answer.lower().split('.') if p.strip()]

        # Count how many phrases have support in sources
        supported = 0
        for phrase in answer_phrases:
            # Simple substring matching (in production, use semantic similarity)
            if len(phrase) > 10 and phrase in source_text:
                supported += 1

        faithfulness = supported / len(answer_phrases) if answer_phrases else 0.0
        logger.debug(f"Answer faithfulness: {faithfulness:.3f}")

        return faithfulness

    def detect_uncertainty_phrases(self, answer: str) -> bool:
        """
        Check if answer contains hedging/uncertainty phrases.

        Interview point: "I detect when the model is uncertain"

        Args:
            answer: Generated answer text

        Returns:
            True if uncertainty detected, False otherwise
        """
        uncertainty_phrases = [
            "i don't know",
            "not sure",
            "unclear",
            "cannot determine",
            "not mentioned",
            "not found in the context",
            "does not provide"
        ]

        answer_lower = answer.lower()
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                logger.debug(f"Uncertainty detected: '{phrase}'")
                return True

        return False

    # ========================================================================
    # CONFIDENCE CALIBRATION
    # ========================================================================

    def add_prediction(
        self,
        confidence: float,
        was_correct: bool
    ):
        """
        Track a prediction for calibration analysis.

        Example:
            System says 80% confident -> Was it actually correct?
            After 100 predictions, do 80% confidence answers succeed 80% of time?

        Interview point: "I track confidence calibration to ensure trust"

        Args:
            confidence: Model's confidence score (0.0 to 1.0)
            was_correct: Whether the answer was actually correct
        """
        # Bucket confidence scores (e.g., 0.7-0.8)
        bucket = int(confidence * 10) / 10  # Round to nearest 0.1
        self.confidence_buckets[bucket].append(1 if was_correct else 0)

        logger.debug(f"Added prediction: confidence={confidence:.2f}, correct={was_correct}")

    def calculate_calibration(self) -> Dict[float, Dict[str, float]]:
        """
        Calculate calibration metrics per confidence bucket.

        Returns:
            Dict mapping confidence bucket to {accuracy, count}

        Example output:
            {
                0.7: {"accuracy": 0.68, "count": 25},  # 70% confidence -> 68% actual
                0.8: {"accuracy": 0.82, "count": 30},  # Well calibrated!
                0.9: {"accuracy": 0.75, "count": 20}   # Overconfident!
            }

        Interview point: "Well-calibrated means 70% confidence = 70% accuracy"
        """
        calibration = {}

        for bucket, outcomes in self.confidence_buckets.items():
            if outcomes:
                accuracy = np.mean(outcomes)
                calibration[bucket] = {
                    "accuracy": accuracy,
                    "count": len(outcomes),
                    "calibration_error": abs(bucket - accuracy)
                }

        logger.info(f"Calibration analysis complete for {len(calibration)} buckets")
        return calibration

    def expected_calibration_error(self) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        Formula: Weighted average of |confidence - accuracy| across buckets

        ECE < 0.05: Well calibrated
        ECE > 0.15: Poorly calibrated

        Interview point: "ECE is a standard metric for confidence calibration"

        Returns:
            ECE score (lower is better)
        """
        calibration = self.calculate_calibration()

        if not calibration:
            return 0.0

        total_samples = sum(data["count"] for data in calibration.values())
        weighted_error = 0.0

        for bucket, data in calibration.items():
            weight = data["count"] / total_samples
            error = data["calibration_error"]
            weighted_error += weight * error

        logger.info(f"Expected Calibration Error: {weighted_error:.3f}")
        return weighted_error

    # ========================================================================
    # COMPREHENSIVE EVALUATION
    # ========================================================================

    def evaluate_query(
        self,
        question: str,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        generated_answer: str,
        source_texts: List[str],
        confidence: float,
        ground_truth_answer: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a single query.

        Interview point: "I measure end-to-end performance with multiple metrics"

        Args:
            question: User's question
            retrieved_docs: Retrieved document IDs
            relevant_docs: Ground truth relevant document IDs
            generated_answer: LLM's generated answer
            source_texts: Text content of source documents
            confidence: System's confidence score
            ground_truth_answer: Optional ground truth answer for comparison

        Returns:
            Dictionary of all evaluation metrics
        """
        metrics = {}

        # Retrieval metrics
        metrics["precision@3"] = self.precision_at_k(retrieved_docs, relevant_docs, k=3)
        metrics["precision@5"] = self.precision_at_k(retrieved_docs, relevant_docs, k=5)
        metrics["recall@3"] = self.recall_at_k(retrieved_docs, relevant_docs, k=3)
        metrics["recall@5"] = self.recall_at_k(retrieved_docs, relevant_docs, k=5)
        metrics["mrr"] = self.mean_reciprocal_rank(retrieved_docs, relevant_docs)

        # Answer quality metrics
        metrics["faithfulness"] = self.answer_faithfulness(generated_answer, source_texts)
        metrics["has_uncertainty"] = self.detect_uncertainty_phrases(generated_answer)

        # Confidence
        metrics["confidence"] = confidence

        # Store for history
        self.metrics_history.append(metrics)

        logger.info(f"Query evaluation complete: P@5={metrics['precision@5']:.3f}, "
                   f"Faithfulness={metrics['faithfulness']:.3f}")

        return metrics

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        Calculate aggregate metrics across all evaluated queries.

        Interview point: "I track system performance over time"

        Returns:
            Dictionary of averaged metrics
        """
        if not self.metrics_history:
            logger.warning("No metrics history available")
            return {}

        # Average numeric metrics
        aggregate = {}
        for key in self.metrics_history[0].keys():
            if key != "has_uncertainty":
                values = [m[key] for m in self.metrics_history if key in m]
                aggregate[f"avg_{key}"] = np.mean(values)
                aggregate[f"std_{key}"] = np.std(values)

        # Count uncertainty occurrences
        uncertainty_count = sum(m["has_uncertainty"] for m in self.metrics_history)
        aggregate["uncertainty_rate"] = uncertainty_count / len(self.metrics_history)

        logger.info(f"Aggregate metrics calculated over {len(self.metrics_history)} queries")
        return aggregate

    # ========================================================================
    # BENCHMARKING
    # ========================================================================

    def compare_configurations(
        self,
        results_baseline: List[Dict],
        results_experimental: List[Dict],
        metric: str = "precision@5"
    ) -> Dict[str, float]:
        """
        Compare two system configurations (A/B testing).

        Example use cases:
        - With vs without reranking
        - Different confidence thresholds
        - Different embedding models

        Interview point: "I validate changes with A/B testing"

        Args:
            results_baseline: Metrics from baseline configuration
            results_experimental: Metrics from experimental configuration
            metric: Which metric to compare

        Returns:
            Comparison statistics
        """
        baseline_values = [r[metric] for r in results_baseline if metric in r]
        experimental_values = [r[metric] for r in results_experimental if metric in r]

        if not baseline_values or not experimental_values:
            logger.warning("Insufficient data for comparison")
            return {}

        comparison = {
            "baseline_mean": np.mean(baseline_values),
            "experimental_mean": np.mean(experimental_values),
            "improvement": np.mean(experimental_values) - np.mean(baseline_values),
            "improvement_pct": (
                (np.mean(experimental_values) - np.mean(baseline_values))
                / np.mean(baseline_values) * 100
            )
        }

        logger.info(f"{metric} comparison: {comparison['improvement_pct']:.1f}% improvement")
        return comparison
