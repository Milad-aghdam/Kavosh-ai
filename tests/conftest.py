"""
Pytest configuration and shared fixtures

This file contains reusable test fixtures that can be used across all test files.

Key Concepts:
- Fixtures: Reusable test data/setup (like mock objects)
- Scope: How long a fixture lasts (function, module, session)
- Mocking: Fake objects that simulate real components
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


# ============================================================================
# MOCK DOCUMENT FIXTURES
# ============================================================================

@pytest.fixture
def mock_document():
    """
    Creates a single mock document with metadata.

    Why we need this:
    - Tests shouldn't depend on real PDFs
    - Fast and repeatable
    - Can control exact content and scores

    Returns:
        Mock object simulating a LangChain Document
    """
    doc = Mock()
    doc.page_content = "This is a sample document about machine learning and neural networks."
    doc.metadata = {
        'relevance_score': 0.75,
        'source': 'test.pdf',
        'page': 1
    }
    return doc


@pytest.fixture
def mock_high_confidence_docs():
    """
    Creates mock documents with HIGH reranker scores.

    Use case: Testing when retrieval is good (should accept answer)
    """
    docs = []
    scores = [0.85, 0.78, 0.72]

    for i, score in enumerate(scores):
        doc = Mock()
        doc.page_content = f"High quality relevant content about the topic. Section {i+1}."
        doc.metadata = {
            'relevance_score': score,
            'source': 'test.pdf',
            'page': i+1
        }
        docs.append(doc)

    return docs


@pytest.fixture
def mock_low_confidence_docs():
    """
    Creates mock documents with LOW reranker scores.

    Use case: Testing when retrieval is poor (should reject answer)
    """
    docs = []
    scores = [0.18, 0.12, 0.08]

    for i, score in enumerate(scores):
        doc = Mock()
        doc.page_content = f"Irrelevant content not related to question. Section {i+1}."
        doc.metadata = {
            'relevance_score': score,
            'source': 'test.pdf',
            'page': i+1
        }
        docs.append(doc)

    return docs


@pytest.fixture
def mock_medium_confidence_docs():
    """
    Creates mock documents with MEDIUM reranker scores.

    Use case: Testing borderline cases (accept with warning)
    """
    docs = []
    scores = [0.55, 0.48, 0.42]

    for i, score in enumerate(scores):
        doc = Mock()
        doc.page_content = f"Partially relevant content. Section {i+1}."
        doc.metadata = {
            'relevance_score': score,
            'source': 'test.pdf',
            'page': i+1
        }
        docs.append(doc)

    return docs


# ============================================================================
# EMBEDDING FIXTURES
# ============================================================================

@pytest.fixture
def sample_question_embedding():
    """
    Sample question embedding for testing semantic similarity.

    Why: Real embeddings are 1024-dimensional vectors from BGE-M3 model
    We use a smaller fixed vector for fast, deterministic tests
    """
    # Normalized random vector simulating an embedding
    vec = np.array([0.5, 0.3, 0.8, 0.1, 0.6])
    return vec / np.linalg.norm(vec)  # Normalize to unit length


@pytest.fixture
def similar_answer_embedding():
    """
    Answer embedding SIMILAR to question (high cosine similarity).

    Use case: Testing when answer is on-topic
    """
    # Similar direction to question
    vec = np.array([0.52, 0.28, 0.82, 0.08, 0.58])
    return vec / np.linalg.norm(vec)


@pytest.fixture
def dissimilar_answer_embedding():
    """
    Answer embedding DIFFERENT from question (low cosine similarity).

    Use case: Testing when answer is off-topic
    """
    # Very different direction
    vec = np.array([0.1, 0.9, 0.05, 0.95, 0.02])
    return vec / np.linalg.norm(vec)


# ============================================================================
# TEXT FIXTURES
# ============================================================================

@pytest.fixture
def sample_pdf_text():
    """
    Sample text simulating PDF extraction.

    Use case: Testing chunking and text processing
    """
    return """
    Machine Learning in Practice

    Machine learning is a subset of artificial intelligence that focuses on
    building systems that can learn from data. Deep learning, a subfield of
    machine learning, uses neural networks with multiple layers.

    The training process involves feeding data through the network and
    adjusting weights based on the error. Common optimizers include SGD,
    Adam, and RMSprop. The learning rate is a crucial hyperparameter.

    Evaluation metrics like accuracy, precision, and recall help assess
    model performance. Cross-validation is used to ensure the model
    generalizes well to unseen data.
    """


@pytest.fixture
def sample_chunks():
    """
    Pre-chunked text for testing retrieval.

    Use case: Testing without needing chunking logic
    """
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "The training process involves feeding data through the network.",
        "Common optimizers include SGD, Adam, and RMSprop.",
        "Evaluation metrics help assess model performance."
    ]


# ============================================================================
# ANSWER TEXT FIXTURES
# ============================================================================

@pytest.fixture
def confident_answer():
    """Answer text that sounds confident."""
    return "The learning rate used in the model is 0.001, as specified in section 3.2."


@pytest.fixture
def uncertain_answer():
    """Answer text expressing uncertainty."""
    return "I don't know the specific learning rate from the provided context."


@pytest.fixture
def partially_uncertain_answer():
    """Answer with hedging phrases."""
    return "The document does not provide information about the exact learning rate, but mentions optimization techniques."


# ============================================================================
# MOCK API RESPONSES
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """
    Mock LLM API response.

    Why: Avoid calling real OpenRouter API in tests (costs money, slow, unreliable)
    """
    response = Mock()
    response.content = "The model uses a learning rate of 0.001 with the Adam optimizer."
    return response


@pytest.fixture
def mock_embedding_model():
    """
    Mock embedding model that returns fixed vectors.

    Why: Avoid loading real BGE-M3 model (slow, memory intensive)
    """
    model = Mock()

    # Return fixed embedding when called
    def mock_embed(text):
        # Different texts get slightly different embeddings
        if "learning rate" in text.lower():
            return np.array([0.5, 0.3, 0.8, 0.1, 0.6])
        else:
            return np.array([0.2, 0.7, 0.1, 0.9, 0.3])

    model.embed_query = Mock(side_effect=mock_embed)
    return model


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def test_config():
    """
    Test configuration without real API keys.

    Why: Tests shouldn't require real credentials
    """
    return {
        "api_key": "test_key_12345",
        "base_url": "https://openrouter.ai/api/v1"
    }


# ============================================================================
# PYTEST MARKERS
# ============================================================================

def pytest_configure(config):
    """
    Register custom markers.

    Usage in tests:
        @pytest.mark.unit
        def test_something():
            ...
    """
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
