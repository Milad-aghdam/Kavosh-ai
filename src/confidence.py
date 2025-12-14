import logging
import numpy as np
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ConfidenceScorer:

    def __init__(
        self,
        reranker_threshold: float = 0.3,
        similarity_threshold: float = 0.5,
        low_confidence_phrases: List[str] = None
    ):

        self.reranker_threshold = reranker_threshold
        self.similarity_threshold = similarity_threshold

        self.low_confidence_phrases = low_confidence_phrases or [
            "i don't know",
            "not found in the context",
            "unclear from the context",
            "cannot determine",
            "not mentioned",
            "does not provide information"
        ]

        logger.info(f"ConfidenceScorer initialized with reranker_threshold={reranker_threshold}")

    def calculate_reranker_confidence(self, documents: List) -> Dict[str, float]:

        if not documents:
            logger.warning("No documents provided for confidence calculation")
            return {
                "max_score": 0.0,
                "avg_score": 0.0,
                "score_gap": 0.0
            }


        scores = []
        for doc in documents:
            score = doc.metadata.get('relevance_score', 0.0)
            scores.append(score)

        if not scores:
            logger.warning("No reranker scores found in document metadata")
            return {
                "max_score": 0.0,
                "avg_score": 0.0,
                "score_gap": 0.0
            }

        scores = sorted(scores, reverse=True)  

        max_score = scores[0]
        avg_score = np.mean(scores)
        score_gap = scores[0] - scores[1] if len(scores) > 1 else scores[0]

        logger.info(f"Reranker confidence - max: {max_score:.3f}, avg: {avg_score:.3f}, gap: {score_gap:.3f}")

        return {
            "max_score": max_score,
            "avg_score": avg_score,
            "score_gap": score_gap
        }

    def calculate_semantic_similarity(
        self,
        question_embedding: np.ndarray,
        answer_embedding: np.ndarray
    ) -> float:

        if question_embedding is None or answer_embedding is None:
            return 0.0

        # Reshape for sklearn (expects 2D arrays)
        q_emb = question_embedding.reshape(1, -1)
        a_emb = answer_embedding.reshape(1, -1)

        # Cosine similarity: 1 = identical direction, 0 = perpendicular, -1 = opposite
        similarity = cosine_similarity(q_emb, a_emb)[0][0]

        logger.info(f"Question-Answer semantic similarity: {similarity:.3f}")

        return float(similarity)

    def check_answer_uncertainty(self, answer_text: str) -> bool:
        answer_lower = answer_text.lower()

        for phrase in self.low_confidence_phrases:
            if phrase in answer_lower:
                logger.info(f"Detected uncertainty phrase: '{phrase}'")
                return True

        return False

    def compute_overall_confidence(
        self,
        reranker_scores: Dict[str, float],
        semantic_similarity: float = None,
        answer_text: str = None
    ) -> Dict[str, any]:
        max_reranker_score = reranker_scores.get("max_score", 0.0)

        # Initialize result
        result = {
            "confidence": 0.0,
            "should_reject": False,
            "rejection_reason": None,
            "confidence_level": "unknown",
            "details": {
                "reranker_max": max_reranker_score,
                "reranker_avg": reranker_scores.get("avg_score", 0.0),
                "score_gap": reranker_scores.get("score_gap", 0.0),
                "semantic_similarity": semantic_similarity,
            }
        }


        if max_reranker_score < self.reranker_threshold:
            result["should_reject"] = True
            result["rejection_reason"] = (
                f"No relevant information found in the document. "
                f"Best match score: {max_reranker_score:.2f} "
                f"(threshold: {self.reranker_threshold:.2f})"
            )
            result["confidence"] = max_reranker_score
            result["confidence_level"] = "low"
            logger.warning(f"Rejecting answer: {result['rejection_reason']}")
            return result

        if answer_text and self.check_answer_uncertainty(answer_text):
            result["should_reject"] = True
            result["rejection_reason"] = (
                "The model could not find a clear answer in the provided context."
            )
            result["confidence"] = max_reranker_score * 0.5  
            result["confidence_level"] = "low"
            logger.warning(f"Rejecting answer: LLM expressed uncertainty")
            return result

        confidence_components = [max_reranker_score]
        weights = [1.0]

        if semantic_similarity is not None:
            confidence_components.append(semantic_similarity)
            weights.append(0.5) 

        overall_confidence = np.average(confidence_components, weights=weights)
        result["confidence"] = float(overall_confidence)

        if overall_confidence >= 0.7:
            result["confidence_level"] = "high"
        elif overall_confidence >= 0.4:
            result["confidence_level"] = "medium"
        else:
            result["confidence_level"] = "low"

        logger.info(
            f"Overall confidence: {overall_confidence:.3f} ({result['confidence_level']})"
        )

        return result

    def format_confidence_message(self, confidence_result: Dict) -> str:

        if confidence_result["should_reject"]:
            return (
                f"⚠️ **Unable to answer confidently**\n\n"
                f"{confidence_result['rejection_reason']}\n\n"
                f"*Suggestion: Try rephrasing your question or check if the topic "
                f"is covered in the uploaded document.*"
            )

        level = confidence_result["confidence_level"]
        score = confidence_result["confidence"]

        if level == "high":
            return f"✅ **High confidence** (score: {score:.2f})"
        elif level == "medium":
            return (
                f"⚠️ **Medium confidence** (score: {score:.2f})\n\n"
                f"*The answer may be partially accurate. "
                f"Please verify with the source passages below.*"
            )
        else:
            return (
                f"⚠️ **Low confidence** (score: {score:.2f})\n\n"
                f"*This answer is uncertain. "
                f"The question may not be fully addressed in the document.*"
            )
