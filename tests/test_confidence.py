"""
Unit tests for confidence scoring module

These tests verify that the ConfidenceScorer class correctly:
- Calculates reranker confidence scores
- Computes semantic similarity
- Detects uncertainty phrases
- Makes correct accept/reject decisions
- Formats user-friendly messages

Interview Keywords: unit testing, test coverage, mocking, assertions
"""

import pytest
import numpy as np
from src.confidence import ConfidenceScorer


# ============================================================================
# TEST CLASS: ConfidenceScorer Initialization
# ============================================================================

@pytest.mark.unit
class TestConfidenceScorerInit:
    """Test ConfidenceScorer initialization and configuration."""

    def test_default_initialization(self):
        """Test that default values are set correctly."""
        scorer = ConfidenceScorer()

        assert scorer.reranker_threshold == 0.3
        assert scorer.similarity_threshold == 0.5
        assert len(scorer.low_confidence_phrases) > 0
        assert "i don't know" in scorer.low_confidence_phrases

    def test_custom_threshold(self):
        """Test initialization with custom thresholds."""
        scorer = ConfidenceScorer(
            reranker_threshold=0.5,
            similarity_threshold=0.7
        )

        assert scorer.reranker_threshold == 0.5
        assert scorer.similarity_threshold == 0.7

    def test_custom_phrases(self):
        """Test initialization with custom uncertainty phrases."""
        custom_phrases = ["uncertain", "maybe", "possibly"]
        scorer = ConfidenceScorer(low_confidence_phrases=custom_phrases)

        assert scorer.low_confidence_phrases == custom_phrases


# ============================================================================
# TEST CLASS: Reranker Confidence Calculation
# ============================================================================

@pytest.mark.unit
class TestRerankerConfidence:
    """Test reranker confidence score extraction and calculation."""

    def test_high_confidence_documents(self, mock_high_confidence_docs):
        """
        Test with high-scoring documents (should accept).

        Interview point: "I test both accept and reject scenarios"
        """
        scorer = ConfidenceScorer(reranker_threshold=0.3)
        result = scorer.calculate_reranker_confidence(mock_high_confidence_docs)

        assert result["max_score"] == 0.85
        assert result["avg_score"] == pytest.approx(0.783, rel=0.01)
        assert result["score_gap"] == pytest.approx(0.07, rel=0.01)

        # Should pass threshold
        assert result["max_score"] > scorer.reranker_threshold

    def test_low_confidence_documents(self, mock_low_confidence_docs):
        """
        Test with low-scoring documents (should reject).

        Interview point: "I validate edge cases and failure modes"
        """
        scorer = ConfidenceScorer(reranker_threshold=0.3)
        result = scorer.calculate_reranker_confidence(mock_low_confidence_docs)

        assert result["max_score"] == 0.18
        assert result["avg_score"] == pytest.approx(0.127, rel=0.01)
        assert result["score_gap"] == pytest.approx(0.06, rel=0.01)

        # Should fail threshold
        assert result["max_score"] < scorer.reranker_threshold

    def test_empty_documents(self):
        """
        Test with no documents (edge case).

        Interview point: "I test boundary conditions"
        """
        scorer = ConfidenceScorer()
        result = scorer.calculate_reranker_confidence([])

        assert result["max_score"] == 0.0
        assert result["avg_score"] == 0.0
        assert result["score_gap"] == 0.0

    def test_single_document(self, mock_document):
        """
        Test with single document (score_gap should equal max_score).

        Interview point: "I handle edge cases correctly"
        """
        scorer = ConfidenceScorer()
        result = scorer.calculate_reranker_confidence([mock_document])

        assert result["max_score"] == 0.75
        assert result["score_gap"] == 0.75  # No second doc to compare

    def test_documents_without_scores(self):
        """
        Test documents missing relevance_score metadata.

        Interview point: "I handle malformed data gracefully"
        """
        from unittest.mock import Mock

        docs = [Mock(metadata={}, page_content="test")]
        scorer = ConfidenceScorer()
        result = scorer.calculate_reranker_confidence(docs)

        # Should default to 0.0
        assert result["max_score"] == 0.0


# ============================================================================
# TEST CLASS: Semantic Similarity Calculation
# ============================================================================

@pytest.mark.unit
class TestSemanticSimilarity:
    """Test semantic similarity calculation between embeddings."""

    def test_identical_embeddings(self, sample_question_embedding):
        """
        Test with identical embeddings (similarity = 1.0).

        Interview point: "I verify mathematical properties"
        """
        scorer = ConfidenceScorer()
        similarity = scorer.calculate_semantic_similarity(
            sample_question_embedding,
            sample_question_embedding
        )

        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_similar_embeddings(
        self,
        sample_question_embedding,
        similar_answer_embedding
    ):
        """
        Test with similar embeddings (high similarity).

        Interview point: "I test expected behavior with realistic data"
        """
        scorer = ConfidenceScorer()
        similarity = scorer.calculate_semantic_similarity(
            sample_question_embedding,
            similar_answer_embedding
        )

        # Should be high but not perfect
        assert 0.9 < similarity < 1.0

    def test_dissimilar_embeddings(
        self,
        sample_question_embedding,
        dissimilar_answer_embedding
    ):
        """
        Test with dissimilar embeddings (low similarity).

        Interview point: "I validate that the system catches off-topic answers"
        """
        scorer = ConfidenceScorer()
        similarity = scorer.calculate_semantic_similarity(
            sample_question_embedding,
            dissimilar_answer_embedding
        )

        # Should be low
        assert similarity < 0.5

    def test_opposite_embeddings(self):
        """
        Test with opposite direction embeddings (similarity = -1.0).

        Interview point: "I test mathematical edge cases"
        """
        scorer = ConfidenceScorer()
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([-1.0, 0.0, 0.0])

        similarity = scorer.calculate_semantic_similarity(emb1, emb2)

        assert similarity == pytest.approx(-1.0, abs=0.01)

    def test_none_embeddings(self):
        """
        Test with None embeddings (should return 0.0).

        Interview point: "I handle null/missing data safely"
        """
        scorer = ConfidenceScorer()

        assert scorer.calculate_semantic_similarity(None, None) == 0.0
        assert scorer.calculate_semantic_similarity(np.array([1.0]), None) == 0.0
        assert scorer.calculate_semantic_similarity(None, np.array([1.0])) == 0.0


# ============================================================================
# TEST CLASS: Answer Uncertainty Detection
# ============================================================================

@pytest.mark.unit
class TestAnswerUncertainty:
    """Test detection of uncertainty phrases in LLM responses."""

    def test_confident_answer(self, confident_answer):
        """
        Test that confident answers are NOT flagged as uncertain.

        Interview point: "I avoid false positives"
        """
        scorer = ConfidenceScorer()
        is_uncertain = scorer.check_answer_uncertainty(confident_answer)

        assert is_uncertain is False

    def test_uncertain_answer(self, uncertain_answer):
        """
        Test that uncertain answers ARE flagged.

        Interview point: "I correctly detect LLM uncertainty"
        """
        scorer = ConfidenceScorer()
        is_uncertain = scorer.check_answer_uncertainty(uncertain_answer)

        assert is_uncertain is True

    def test_partial_uncertainty(self, partially_uncertain_answer):
        """
        Test detection of hedging phrases.

        Interview point: "I catch subtle uncertainty signals"
        """
        scorer = ConfidenceScorer()
        is_uncertain = scorer.check_answer_uncertainty(partially_uncertain_answer)

        assert is_uncertain is True

    def test_case_insensitive(self):
        """
        Test that detection is case-insensitive.

        Interview point: "I handle text variations robustly"
        """
        scorer = ConfidenceScorer()

        assert scorer.check_answer_uncertainty("I DON'T KNOW") is True
        assert scorer.check_answer_uncertainty("i Don't KnOw") is True

    def test_custom_phrases(self):
        """
        Test with custom uncertainty phrases.

        Interview point: "My system is configurable"
        """
        scorer = ConfidenceScorer(low_confidence_phrases=["uncertain", "maybe"])

        assert scorer.check_answer_uncertainty("I'm uncertain about this") is True
        assert scorer.check_answer_uncertainty("Maybe it works") is True
        assert scorer.check_answer_uncertainty("I don't know") is False  # Not in custom list


# ============================================================================
# TEST CLASS: Overall Confidence Computation
# ============================================================================

@pytest.mark.unit
class TestOverallConfidence:
    """Test the overall confidence computation and decision logic."""

    def test_high_confidence_scenario(self):
        """
        Test high reranker + high similarity = high confidence (accept).

        Interview point: "I test the happy path"
        """
        scorer = ConfidenceScorer(reranker_threshold=0.3)
        reranker_scores = {"max_score": 0.85, "avg_score": 0.75, "score_gap": 0.10}

        result = scorer.compute_overall_confidence(
            reranker_scores=reranker_scores,
            semantic_similarity=0.82,
            answer_text="The learning rate is 0.001."
        )

        assert result["should_reject"] is False
        assert result["confidence_level"] == "high"
        assert result["confidence"] > 0.7

    def test_low_reranker_rejection(self):
        """
        Test low reranker score triggers rejection.

        Interview point: "I validate early rejection logic"
        """
        scorer = ConfidenceScorer(reranker_threshold=0.3)
        reranker_scores = {"max_score": 0.15, "avg_score": 0.12, "score_gap": 0.03}

        result = scorer.compute_overall_confidence(
            reranker_scores=reranker_scores,
            semantic_similarity=0.70,  # Even with good similarity
            answer_text="Some answer"
        )

        assert result["should_reject"] is True
        assert result["confidence_level"] == "low"
        assert "No relevant information" in result["rejection_reason"]

    def test_llm_uncertainty_rejection(self):
        """
        Test LLM expressing uncertainty triggers rejection.

        Interview point: "I use LLM self-assessment"
        """
        scorer = ConfidenceScorer(reranker_threshold=0.3)
        reranker_scores = {"max_score": 0.65, "avg_score": 0.55, "score_gap": 0.10}

        result = scorer.compute_overall_confidence(
            reranker_scores=reranker_scores,
            semantic_similarity=0.60,
            answer_text="I don't know the answer from the context."
        )

        assert result["should_reject"] is True
        assert result["confidence_level"] == "low"
        assert "could not find a clear answer" in result["rejection_reason"]

    def test_medium_confidence(self):
        """
        Test medium confidence scenario (accept with warning).

        Interview point: "I handle borderline cases"
        """
        scorer = ConfidenceScorer(reranker_threshold=0.3)
        reranker_scores = {"max_score": 0.55, "avg_score": 0.48, "score_gap": 0.07}

        result = scorer.compute_overall_confidence(
            reranker_scores=reranker_scores,
            semantic_similarity=0.50,
            answer_text="The paper discusses optimization."
        )

        assert result["should_reject"] is False
        assert result["confidence_level"] == "medium"
        assert 0.4 <= result["confidence"] < 0.7

    def test_without_semantic_similarity(self):
        """
        Test confidence calculation without semantic similarity.

        Interview point: "I handle optional parameters gracefully"
        """
        scorer = ConfidenceScorer(reranker_threshold=0.3)
        reranker_scores = {"max_score": 0.75, "avg_score": 0.65, "score_gap": 0.10}

        result = scorer.compute_overall_confidence(
            reranker_scores=reranker_scores,
            semantic_similarity=None,
            answer_text="Some answer"
        )

        # Should still work, using only reranker score
        assert result["should_reject"] is False
        assert result["confidence"] == 0.75

    def test_confidence_details(self):
        """
        Test that confidence details are properly populated.

        Interview point: "I provide transparency in scoring"
        """
        scorer = ConfidenceScorer()
        reranker_scores = {"max_score": 0.80, "avg_score": 0.70, "score_gap": 0.10}

        result = scorer.compute_overall_confidence(
            reranker_scores=reranker_scores,
            semantic_similarity=0.75
        )

        assert "details" in result
        assert result["details"]["reranker_max"] == 0.80
        assert result["details"]["reranker_avg"] == 0.70
        assert result["details"]["semantic_similarity"] == 0.75


# ============================================================================
# TEST CLASS: Confidence Message Formatting
# ============================================================================

@pytest.mark.unit
class TestConfidenceMessages:
    """Test user-facing confidence message formatting."""

    def test_rejection_message(self):
        """Test rejection message format."""
        scorer = ConfidenceScorer()
        result = {
            "should_reject": True,
            "rejection_reason": "Test rejection",
            "confidence": 0.15,
            "confidence_level": "low"
        }

        message = scorer.format_confidence_message(result)

        assert "Unable to answer confidently" in message
        assert "Test rejection" in message
        assert "Suggestion" in message

    def test_high_confidence_message(self):
        """Test high confidence message format."""
        scorer = ConfidenceScorer()
        result = {
            "should_reject": False,
            "confidence": 0.85,
            "confidence_level": "high"
        }

        message = scorer.format_confidence_message(result)

        assert "High confidence" in message
        assert "0.85" in message

    def test_medium_confidence_message(self):
        """Test medium confidence warning message."""
        scorer = ConfidenceScorer()
        result = {
            "should_reject": False,
            "confidence": 0.55,
            "confidence_level": "medium"
        }

        message = scorer.format_confidence_message(result)

        assert "Medium confidence" in message
        assert "may be partially accurate" in message
        assert "verify" in message.lower()

    def test_low_confidence_message(self):
        """Test low confidence (but not rejected) message."""
        scorer = ConfidenceScorer()
        result = {
            "should_reject": False,
            "confidence": 0.35,
            "confidence_level": "low"
        }

        message = scorer.format_confidence_message(result)

        assert "Low confidence" in message
        assert "uncertain" in message.lower()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestConfidenceScorerIntegration:
    """Integration tests combining multiple confidence signals."""

    def test_full_pipeline_accept(
        self,
        mock_high_confidence_docs,
        sample_question_embedding,
        similar_answer_embedding,
        confident_answer
    ):
        """
        Test full confidence scoring pipeline - accept scenario.

        Interview point: "I test end-to-end workflows"
        """
        scorer = ConfidenceScorer(reranker_threshold=0.3)

        # Step 1: Calculate reranker confidence
        reranker_conf = scorer.calculate_reranker_confidence(mock_high_confidence_docs)

        # Step 2: Calculate semantic similarity
        similarity = scorer.calculate_semantic_similarity(
            sample_question_embedding,
            similar_answer_embedding
        )

        # Step 3: Compute overall confidence
        result = scorer.compute_overall_confidence(
            reranker_scores=reranker_conf,
            semantic_similarity=similarity,
            answer_text=confident_answer
        )

        # Assertions
        assert result["should_reject"] is False
        assert result["confidence_level"] == "high"
        assert result["confidence"] > 0.7

    def test_full_pipeline_reject(
        self,
        mock_low_confidence_docs,
        sample_question_embedding,
        dissimilar_answer_embedding,
        uncertain_answer
    ):
        """
        Test full confidence scoring pipeline - reject scenario.

        Interview point: "I validate rejection paths end-to-end"
        """
        scorer = ConfidenceScorer(reranker_threshold=0.3)

        # Step 1: Calculate reranker confidence
        reranker_conf = scorer.calculate_reranker_confidence(mock_low_confidence_docs)

        # Should reject based on reranker alone
        result = scorer.compute_overall_confidence(
            reranker_scores=reranker_conf,
            semantic_similarity=None,
            answer_text=None
        )

        assert result["should_reject"] is True
        assert result["confidence_level"] == "low"
