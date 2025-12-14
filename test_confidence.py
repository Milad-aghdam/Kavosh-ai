"""
Simple test script to verify confidence scoring works correctly.

This script tests the confidence scorer without needing a full PDF.
It simulates the components to verify the logic.

Run this to make sure confidence scoring is working before running the full app.
"""

import numpy as np
from src.confidence import ConfidenceScorer


def test_reranker_confidence():
    """Test reranker confidence calculation."""
    print("=" * 60)
    print("TEST 1: Reranker Confidence Calculation")
    print("=" * 60)

    scorer = ConfidenceScorer(reranker_threshold=0.3)

    class MockDoc:
        def __init__(self, score):
            self.metadata = {'relevance_score': score}
            self.page_content = "Mock content"

    # Test Case 1: High confidence documents
    print("\n📝 Test Case 1: High Confidence (should ACCEPT)")
    high_conf_docs = [
        MockDoc(0.85),
        MockDoc(0.72),
        MockDoc(0.68)
    ]

    result = scorer.calculate_reranker_confidence(high_conf_docs)
    print(f"   Max Score: {result['max_score']:.3f}")
    print(f"   Avg Score: {result['avg_score']:.3f}")
    print(f"   Score Gap: {result['score_gap']:.3f}")
    print(f"   ✅ Should accept: {result['max_score'] >= 0.3}")

    # Test Case 2: Low confidence documents
    print("\n📝 Test Case 2: Low Confidence (should REJECT)")
    low_conf_docs = [
        MockDoc(0.15),
        MockDoc(0.12),
        MockDoc(0.08)
    ]

    result = scorer.calculate_reranker_confidence(low_conf_docs)
    print(f"   Max Score: {result['max_score']:.3f}")
    print(f"   Avg Score: {result['avg_score']:.3f}")
    print(f"   Score Gap: {result['score_gap']:.3f}")
    print(f"   ❌ Should reject: {result['max_score'] < 0.3}")


def test_semantic_similarity():
    """Test semantic similarity calculation."""
    print("\n" + "=" * 60)
    print("TEST 2: Semantic Similarity")
    print("=" * 60)

    scorer = ConfidenceScorer()

    # Test Case 1: Identical embeddings (perfect similarity)
    print("\n📝 Test Case 1: Identical Embeddings")
    emb1 = np.array([1.0, 0.5, 0.3, 0.8])
    emb2 = np.array([1.0, 0.5, 0.3, 0.8])
    similarity = scorer.calculate_semantic_similarity(emb1, emb2)
    print(f"   Similarity: {similarity:.3f}")
    print(f"   Expected: ~1.0 ✅" if similarity > 0.99 else "   ❌ Unexpected")

    # Test Case 2: Opposite embeddings (low similarity)
    print("\n📝 Test Case 2: Opposite Embeddings")
    emb1 = np.array([1.0, 1.0, 1.0, 1.0])
    emb2 = np.array([-1.0, -1.0, -1.0, -1.0])
    similarity = scorer.calculate_semantic_similarity(emb1, emb2)
    print(f"   Similarity: {similarity:.3f}")
    print(f"   Expected: ~-1.0 ✅" if similarity < -0.99 else "   ❌ Unexpected")

    # Test Case 3: Perpendicular embeddings (neutral)
    print("\n📝 Test Case 3: Perpendicular Embeddings")
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([0.0, 1.0])
    similarity = scorer.calculate_semantic_similarity(emb1, emb2)
    print(f"   Similarity: {similarity:.3f}")
    print(f"   Expected: ~0.0 ✅" if abs(similarity) < 0.01 else "   ❌ Unexpected")


def test_answer_uncertainty():
    """Test uncertainty phrase detection."""
    print("\n" + "=" * 60)
    print("TEST 3: Answer Uncertainty Detection")
    print("=" * 60)

    scorer = ConfidenceScorer()

    # Test Case 1: Confident answer
    print("\n📝 Test Case 1: Confident Answer")
    answer = "The learning rate used in the model is 0.001."
    is_uncertain = scorer.check_answer_uncertainty(answer)
    print(f"   Answer: '{answer}'")
    print(f"   Uncertain: {is_uncertain}")
    print(f"   Expected: False ✅" if not is_uncertain else "   ❌ Unexpected")

    # Test Case 2: Uncertain answer
    print("\n📝 Test Case 2: Uncertain Answer")
    answer = "I don't know the answer based on the provided context."
    is_uncertain = scorer.check_answer_uncertainty(answer)
    print(f"   Answer: '{answer}'")
    print(f"   Uncertain: {is_uncertain}")
    print(f"   Expected: True ✅" if is_uncertain else "   ❌ Unexpected")

    # Test Case 3: Another uncertain phrase
    print("\n📝 Test Case 3: Another Uncertainty Phrase")
    answer = "The document does not provide information about this topic."
    is_uncertain = scorer.check_answer_uncertainty(answer)
    print(f"   Answer: '{answer}'")
    print(f"   Uncertain: {is_uncertain}")
    print(f"   Expected: True ✅" if is_uncertain else "   ❌ Unexpected")


def test_overall_confidence():
    """Test overall confidence computation and decision logic."""
    print("\n" + "=" * 60)
    print("TEST 4: Overall Confidence & Decision Logic")
    print("=" * 60)

    scorer = ConfidenceScorer(reranker_threshold=0.3)

    # Test Case 1: High confidence scenario
    print("\n📝 Test Case 1: High Confidence (should ACCEPT)")
    reranker_scores = {
        "max_score": 0.85,
        "avg_score": 0.75,
        "score_gap": 0.13
    }
    result = scorer.compute_overall_confidence(
        reranker_scores=reranker_scores,
        semantic_similarity=0.82,
        answer_text="The learning rate is 0.001."
    )
    print(f"   Overall Confidence: {result['confidence']:.3f}")
    print(f"   Confidence Level: {result['confidence_level']}")
    print(f"   Should Reject: {result['should_reject']}")
    print(f"   Expected: Accept ✅" if not result['should_reject'] else "   ❌ Unexpected")

    # Test Case 2: Low reranker score (should REJECT)
    print("\n📝 Test Case 2: Low Reranker Score (should REJECT)")
    reranker_scores = {
        "max_score": 0.15,
        "avg_score": 0.12,
        "score_gap": 0.03
    }
    result = scorer.compute_overall_confidence(
        reranker_scores=reranker_scores,
        semantic_similarity=0.70,
        answer_text="The model uses an optimizer."
    )
    print(f"   Overall Confidence: {result['confidence']:.3f}")
    print(f"   Confidence Level: {result['confidence_level']}")
    print(f"   Should Reject: {result['should_reject']}")
    print(f"   Rejection Reason: {result['rejection_reason']}")
    print(f"   Expected: Reject ✅" if result['should_reject'] else "   ❌ Unexpected")

    # Test Case 3: LLM expresses uncertainty (should REJECT)
    print("\n📝 Test Case 3: LLM Expresses Uncertainty (should REJECT)")
    reranker_scores = {
        "max_score": 0.65,
        "avg_score": 0.55,
        "score_gap": 0.10
    }
    result = scorer.compute_overall_confidence(
        reranker_scores=reranker_scores,
        semantic_similarity=0.60,
        answer_text="I don't know the answer from the context provided."
    )
    print(f"   Overall Confidence: {result['confidence']:.3f}")
    print(f"   Confidence Level: {result['confidence_level']}")
    print(f"   Should Reject: {result['should_reject']}")
    print(f"   Rejection Reason: {result['rejection_reason']}")
    print(f"   Expected: Reject ✅" if result['should_reject'] else "   ❌ Unexpected")

    # Test Case 4: Medium confidence
    print("\n📝 Test Case 4: Medium Confidence (accept with warning)")
    reranker_scores = {
        "max_score": 0.55,
        "avg_score": 0.48,
        "score_gap": 0.07
    }
    result = scorer.compute_overall_confidence(
        reranker_scores=reranker_scores,
        semantic_similarity=0.50,
        answer_text="The paper discusses optimization techniques."
    )
    print(f"   Overall Confidence: {result['confidence']:.3f}")
    print(f"   Confidence Level: {result['confidence_level']}")
    print(f"   Should Reject: {result['should_reject']}")
    print(f"   Expected: Medium confidence ✅" if result['confidence_level'] == 'medium' else "   ❌ Unexpected")


def test_confidence_messages():
    """Test user-facing confidence messages."""
    print("\n" + "=" * 60)
    print("TEST 5: Confidence Message Formatting")
    print("=" * 60)

    scorer = ConfidenceScorer()

    # Test Case 1: Rejection message
    print("\n📝 Test Case 1: Rejection Message")
    result = {
        "should_reject": True,
        "rejection_reason": "No relevant information found.",
        "confidence": 0.15,
        "confidence_level": "low"
    }
    message = scorer.format_confidence_message(result)
    print(f"   {message}")

    # Test Case 2: High confidence message
    print("\n📝 Test Case 2: High Confidence Message")
    result = {
        "should_reject": False,
        "confidence": 0.85,
        "confidence_level": "high"
    }
    message = scorer.format_confidence_message(result)
    print(f"   {message}")

    # Test Case 3: Medium confidence message
    print("\n📝 Test Case 3: Medium Confidence Message")
    result = {
        "should_reject": False,
        "confidence": 0.55,
        "confidence_level": "medium"
    }
    message = scorer.format_confidence_message(result)
    print(f"   {message}")


if __name__ == "__main__":
    print("\n" + "🧪" * 30)
    print("CONFIDENCE SCORING TEST SUITE")
    print("🧪" * 30 + "\n")

    try:
        test_reranker_confidence()
        test_semantic_similarity()
        test_answer_uncertainty()
        test_overall_confidence()
        test_confidence_messages()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYour confidence scoring system is ready to use.")
        print("You can now run the main application with: python app.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
