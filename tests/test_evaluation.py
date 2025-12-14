
import pytest
import numpy as np
from src.evaluation import RAGEvaluator

@pytest.mark.unit
class TestRetrievalMetrics:
    

    def test_precision_at_k_perfect(self):
        
        evaluator = RAGEvaluator()

        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        precision = evaluator.precision_at_k(retrieved, relevant, k=5)

        assert precision == 1.0

    def test_precision_at_k_partial(self):
        evaluator = RAGEvaluator()

        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc5"]

        precision = evaluator.precision_at_k(retrieved, relevant, k=5)

        # 3 out of 5 are relevant
        assert precision == 0.6

    def test_precision_at_k_zero(self):
        """Test precision@k with no relevant docs."""
        evaluator = RAGEvaluator()

        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4", "doc5", "doc6"]

        precision = evaluator.precision_at_k(retrieved, relevant, k=3)

        assert precision == 0.0

    def test_precision_at_k_empty_retrieved(self):
       
        evaluator = RAGEvaluator()

        precision = evaluator.precision_at_k([], ["doc1", "doc2"], k=5)

        assert precision == 0.0

    def test_recall_at_k_perfect(self):
        """Test recall@k with all relevant docs retrieved."""
        evaluator = RAGEvaluator()

        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc2", "doc3"]

        recall = evaluator.recall_at_k(retrieved, relevant, k=5)

        assert recall == 1.0

    def test_recall_at_k_partial(self):
       
        evaluator = RAGEvaluator()

        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc6", "doc7"]

        recall = evaluator.recall_at_k(retrieved, relevant, k=5)

        # Found 2 out of 4 relevant docs
        assert recall == 0.5

    def test_recall_at_k_empty_relevant(self):
        """Test recall@k when there are no relevant docs."""
        evaluator = RAGEvaluator()

        recall = evaluator.recall_at_k(["doc1", "doc2"], [], k=5)

        assert recall == 0.0

    def test_mrr_first_position(self):
      
        evaluator = RAGEvaluator()

        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc5"]

        mrr = evaluator.mean_reciprocal_rank(retrieved, relevant)

        assert mrr == 1.0

    def test_mrr_third_position(self):
        evaluator = RAGEvaluator()

        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = ["doc3", "doc5"]

        mrr = evaluator.mean_reciprocal_rank(retrieved, relevant)

        # First relevant at position 3: MRR = 1/3
        assert mrr == pytest.approx(0.333, rel=0.01)

    def test_mrr_no_relevant(self):

        evaluator = RAGEvaluator()

        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4", "doc5"]

        mrr = evaluator.mean_reciprocal_rank(retrieved, relevant)

        assert mrr == 0.0



@pytest.mark.unit
class TestAnswerQuality:
    

    def test_faithfulness_perfect(self):

        evaluator = RAGEvaluator()

        answer = "In our experiments we use a learning rate of 0.001 with the Adam optimizer. We set the batch size to 32 for all training runs."
        sources = [
            "In our experiments, we use a learning rate of 0.001 with the Adam optimizer.",
            "We set the batch size to 32 for all training runs."
        ]

        faithfulness = evaluator.answer_faithfulness(answer, sources)

        # Should be high (simple substring matching)
        assert faithfulness > 0.3

    def test_faithfulness_hallucinated(self):
        
        evaluator = RAGEvaluator()

        answer = "The model was trained on 1 billion examples for 3 months."
        sources = [
            "The dataset contains 10,000 examples.",
            "Training took approximately 2 hours on a V100 GPU."
        ]

        faithfulness = evaluator.answer_faithfulness(answer, sources)

        assert faithfulness < 0.5

    def test_faithfulness_empty_answer(self):
        """Test faithfulness with empty answer."""
        evaluator = RAGEvaluator()

        faithfulness = evaluator.answer_faithfulness("", ["source1", "source2"])

        assert faithfulness == 0.0

    def test_faithfulness_empty_sources(self):
        """Test faithfulness with no sources."""
        evaluator = RAGEvaluator()

        faithfulness = evaluator.answer_faithfulness("Some answer", [])

        assert faithfulness == 0.0

    def test_detect_uncertainty_explicit(self):

        evaluator = RAGEvaluator()

        uncertain_answers = [
            "I don't know the answer from the context.",
            "This is not mentioned in the document.",
            "The context does not provide information about this.",
            "I cannot determine the answer."
        ]

        for answer in uncertain_answers:
            assert evaluator.detect_uncertainty_phrases(answer) is True

    def test_detect_uncertainty_confident(self):
        """Test that confident answers aren't flagged as uncertain."""
        evaluator = RAGEvaluator()

        confident_answer = "The learning rate is 0.001, as stated in section 3.2."

        assert evaluator.detect_uncertainty_phrases(confident_answer) is False

    def test_detect_uncertainty_case_insensitive(self):
        """Test that detection works regardless of case."""
        evaluator = RAGEvaluator()

        assert evaluator.detect_uncertainty_phrases("I DON'T KNOW") is True
        assert evaluator.detect_uncertainty_phrases("i don't know") is True
        assert evaluator.detect_uncertainty_phrases("I Don't Know") is True


# ============================================================================
# TEST CLASS: Confidence Calibration
# ============================================================================

@pytest.mark.unit
class TestConfidenceCalibration:
    """Test confidence calibration tracking and analysis."""

    def test_add_prediction(self):
        """
        Test adding predictions for calibration tracking.

        Interview point: "I track predictions to measure calibration"
        """
        evaluator = RAGEvaluator()

        evaluator.add_prediction(confidence=0.8, was_correct=True)
        evaluator.add_prediction(confidence=0.8, was_correct=True)
        evaluator.add_prediction(confidence=0.8, was_correct=False)

        # Should have 3 predictions in 0.8 bucket
        assert len(evaluator.confidence_buckets[0.8]) == 3

    def test_calculate_calibration_perfect(self):
        """
        Test calibration calculation with perfect calibration.

        Interview point: "Perfect calibration means confidence = accuracy"
        """
        evaluator = RAGEvaluator()

        # Add perfectly calibrated predictions
        # 70% confidence -> 70% accuracy
        for i in range(7):
            evaluator.add_prediction(0.7, True)
        for i in range(3):
            evaluator.add_prediction(0.7, False)

        calibration = evaluator.calculate_calibration()

        assert 0.7 in calibration
        assert calibration[0.7]["accuracy"] == pytest.approx(0.7, abs=0.01)
        assert calibration[0.7]["count"] == 10
        assert calibration[0.7]["calibration_error"] < 0.05

    def test_calculate_calibration_overconfident(self):
        """
        Test detection of overconfidence.

        Interview point: "Overconfidence is dangerous - 90% confidence but 60% accuracy"
        """
        evaluator = RAGEvaluator()

        # System says 90% confident but only 60% accurate
        for i in range(6):
            evaluator.add_prediction(0.9, True)
        for i in range(4):
            evaluator.add_prediction(0.9, False)

        calibration = evaluator.calculate_calibration()

        assert calibration[0.9]["accuracy"] == pytest.approx(0.6, abs=0.01)
        assert calibration[0.9]["calibration_error"] > 0.25  # |0.9 - 0.6| = 0.3

    def test_expected_calibration_error_perfect(self):
        """
        Test ECE calculation with perfect calibration.

        Interview point: "ECE < 0.05 means well-calibrated system"
        """
        evaluator = RAGEvaluator()

        # Add perfectly calibrated data at multiple levels
        for conf in [0.5, 0.7, 0.9]:
            num_correct = int(conf * 10)
            num_incorrect = 10 - num_correct

            for i in range(num_correct):
                evaluator.add_prediction(conf, True)
            for i in range(num_incorrect):
                evaluator.add_prediction(conf, False)

        ece = evaluator.expected_calibration_error()

        # Should be very low
        assert ece < 0.05

    def test_expected_calibration_error_poor(self):
        """Test ECE with poorly calibrated predictions."""
        evaluator = RAGEvaluator()

        # Very overconfident predictions
        for i in range(10):
            evaluator.add_prediction(0.9, False)  # 90% confident but all wrong!

        ece = evaluator.expected_calibration_error()

        # Should be very high
        assert ece > 0.5

    def test_calibration_empty(self):
        """
        Test calibration with no predictions.

        Interview point: "I handle empty data gracefully"
        """
        evaluator = RAGEvaluator()

        calibration = evaluator.calculate_calibration()
        ece = evaluator.expected_calibration_error()

        assert calibration == {}
        assert ece == 0.0


# ============================================================================
# TEST CLASS: Comprehensive Evaluation
# ============================================================================

@pytest.mark.unit
class TestComprehensiveEvaluation:
    """Test end-to-end query evaluation."""

    def test_evaluate_query_complete(self):
        """
        Test comprehensive query evaluation with all metrics.

        Interview point: "I measure multiple aspects of quality"
        """
        evaluator = RAGEvaluator()

        metrics = evaluator.evaluate_query(
            question="What is the learning rate?",
            retrieved_docs=["doc1", "doc2", "doc3"],
            relevant_docs=["doc1", "doc3"],
            generated_answer="The learning rate is 0.001.",
            source_texts=["We use learning rate 0.001 in experiments."],
            confidence=0.85,
            ground_truth_answer="0.001"
        )

        # Should have all expected metrics
        assert "precision@3" in metrics
        assert "precision@5" in metrics
        assert "recall@3" in metrics
        assert "recall@5" in metrics
        assert "mrr" in metrics
        assert "faithfulness" in metrics
        assert "has_uncertainty" in metrics
        assert "confidence" in metrics

        # Check metric values
        assert metrics["precision@3"] == pytest.approx(0.666, rel=0.01)
        assert metrics["confidence"] == 0.85

    def test_get_aggregate_metrics(self):
        """
        Test aggregate metrics calculation over multiple queries.

        Interview point: "I track system performance over time"
        """
        evaluator = RAGEvaluator()

        # Evaluate multiple queries
        for i in range(5):
            evaluator.evaluate_query(
                question=f"Question {i}",
                retrieved_docs=["doc1", "doc2"],
                relevant_docs=["doc1"],
                generated_answer="Answer",
                source_texts=["Source"],
                confidence=0.7
            )

        aggregate = evaluator.get_aggregate_metrics()

        # Should have averaged metrics
        assert "avg_precision@3" in aggregate
        assert "avg_confidence" in aggregate
        assert "std_precision@3" in aggregate
        assert "uncertainty_rate" in aggregate

        # Values should be reasonable
        assert 0 <= aggregate["avg_confidence"] <= 1
        assert 0 <= aggregate["uncertainty_rate"] <= 1


# ============================================================================
# TEST CLASS: Benchmarking
# ============================================================================

@pytest.mark.unit
class TestBenchmarking:
    """Test A/B testing and configuration comparison."""

    def test_compare_configurations_improvement(self):
        """
        Test comparison showing improvement.

        Interview point: "I validate changes with A/B testing"
        """
        evaluator = RAGEvaluator()

        baseline = [{"precision@5": 0.5}, {"precision@5": 0.6}]
        experimental = [{"precision@5": 0.7}, {"precision@5": 0.8}]

        comparison = evaluator.compare_configurations(
            baseline,
            experimental,
            metric="precision@5"
        )

        assert comparison["baseline_mean"] == pytest.approx(0.55, abs=0.01)
        assert comparison["experimental_mean"] == pytest.approx(0.75, abs=0.01)
        assert comparison["improvement"] > 0
        assert comparison["improvement_pct"] > 0

    def test_compare_configurations_regression(self):
        """
        Test comparison showing regression.

        Interview point: "I catch performance regressions before deployment"
        """
        evaluator = RAGEvaluator()

        baseline = [{"precision@5": 0.8}, {"precision@5": 0.7}]
        experimental = [{"precision@5": 0.5}, {"precision@5": 0.6}]

        comparison = evaluator.compare_configurations(
            baseline,
            experimental,
            metric="precision@5"
        )

        # Should show negative improvement (regression)
        assert comparison["improvement"] < 0
        assert comparison["improvement_pct"] < 0

    def test_compare_configurations_empty(self):
        """Test comparison with empty results."""
        evaluator = RAGEvaluator()

        comparison = evaluator.compare_configurations([], [], metric="precision@5")

        assert comparison == {}


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestEvaluationIntegration:
    """Integration tests for full evaluation workflow."""

    def test_full_evaluation_workflow(self):
        """
        Test complete evaluation workflow from query to calibration.

        Interview point: "I test end-to-end workflows"
        """
        evaluator = RAGEvaluator()

        # Simulate evaluating 10 queries
        for i in range(10):
            metrics = evaluator.evaluate_query(
                question=f"Question {i}",
                retrieved_docs=[f"doc{j}" for j in range(5)],
                relevant_docs=[f"doc{i % 3}"],
                generated_answer=f"Answer {i}",
                source_texts=[f"Source {i}"],
                confidence=0.7 + (i % 3) * 0.1
            )

            # Track for calibration
            is_correct = metrics["precision@5"] > 0.1
            evaluator.add_prediction(metrics["confidence"], is_correct)

        # Get aggregate metrics
        aggregate = evaluator.get_aggregate_metrics()

        # Get calibration
        calibration = evaluator.calculate_calibration()
        ece = evaluator.expected_calibration_error()

        # Verify we have complete results
        assert len(evaluator.metrics_history) == 10
        assert len(aggregate) > 5
        assert len(calibration) > 0
        assert ece >= 0.0
