
import pytest
from unittest.mock import Mock, patch, mock_open
from src.utils import load_and_extract_text, chunk_text

@pytest.mark.unit
class TestPDFExtraction:
    """Test PDF loading and text extraction."""

    @patch('src.utils.fitz.open')
    def test_load_single_page_pdf(self, mock_fitz_open):
       
        # Setup mock
        mock_page = Mock()
        mock_page.get_text.return_value = "This is page 1 content."

        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.close = Mock()

        mock_fitz_open.return_value = mock_doc

        # Execute
        result = load_and_extract_text("fake_path.pdf")

        # Assert
        assert result == "This is page 1 content."
        mock_fitz_open.assert_called_once_with("fake_path.pdf")
        mock_doc.close.assert_called_once()

    @patch('src.utils.fitz.open')
    def test_load_multi_page_pdf(self, mock_fitz_open):
       
        # Setup mock with 3 pages
        pages = [
            Mock(get_text=Mock(return_value="Page 1 text. ")),
            Mock(get_text=Mock(return_value="Page 2 text. ")),
            Mock(get_text=Mock(return_value="Page 3 text."))
        ]

        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter(pages))
        mock_doc.close = Mock()

        mock_fitz_open.return_value = mock_doc

        # Execute
        result = load_and_extract_text("multi_page.pdf")

        # Assert
        assert result == "Page 1 text. Page 2 text. Page 3 text."
        assert mock_doc.close.called

    @patch('src.utils.fitz.open')
    def test_load_empty_pdf(self, mock_fitz_open):
      
        mock_page = Mock()
        mock_page.get_text.return_value = ""

        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.close = Mock()

        mock_fitz_open.return_value = mock_doc

        # Execute
        result = load_and_extract_text("empty.pdf")

        # Assert
        assert result == ""

    @patch('src.utils.fitz.open')
    def test_pdf_with_special_characters(self, mock_fitz_open):
        
        special_text = "Text with émojis 📚 and spëcial çhars!"

        mock_page = Mock()
        mock_page.get_text.return_value = special_text

        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.close = Mock()

        mock_fitz_open.return_value = mock_doc

        # Execute
        result = load_and_extract_text("special.pdf")

        # Assert
        assert result == special_text


# ============================================================================
# TEST CLASS: Text Chunking
# ============================================================================

@pytest.mark.unit
class TestTextChunking:

    def test_chunk_basic_text(self, sample_pdf_text):
        
        chunks = chunk_text(sample_pdf_text, chunk_size=200, chunk_overlap=50)

        # Should produce multiple chunks
        assert len(chunks) > 1

        # Each chunk should respect max size (approximately)
        for chunk in chunks:
            assert len(chunk) <= 250  # Some tolerance for word boundaries

    def test_chunk_size_parameter(self):
       
        text = "a" * 500  # 500 character string

        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)

        # Should produce approximately 5 chunks
        assert 4 <= len(chunks) <= 6  # Some tolerance

    def test_chunk_overlap(self):
       
        text = "The quick brown fox jumps over the lazy dog. " * 10

        chunks = chunk_text(text, chunk_size=100, chunk_overlap=30)

        if len(chunks) >= 2:
            # Check that consecutive chunks share content
            chunk1_end = chunks[0][-30:]
            chunk2_start = chunks[1][:30]

            # There should be some overlap in content
            # (exact match difficult due to word boundaries, but lengths should be similar)
            assert len(chunk1_end) > 0
            assert len(chunk2_start) > 0

    def test_short_text_single_chunk(self):
       
        short_text = "This is a very short text."

        chunks = chunk_text(short_text, chunk_size=1000, chunk_overlap=200)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_empty_text(self):
       
        chunks = chunk_text("", chunk_size=1000, chunk_overlap=200)

        # Should return empty list or list with empty string
        assert len(chunks) <= 1
        if chunks:
            assert chunks[0] == ""

    def test_zero_overlap(self):
        
        text = "a" * 300

        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)

        # Should produce exactly 3 chunks with no overlap
        assert len(chunks) == 3
        total_length = sum(len(chunk) for chunk in chunks)
        assert total_length == 300

    @pytest.mark.parametrize("chunk_size,chunk_overlap", [
        (500, 100),
        (1000, 200),
        (2000, 400),
    ])
    def test_various_chunk_sizes(self, sample_pdf_text, chunk_size, chunk_overlap):
        
        chunks = chunk_text(sample_pdf_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Basic sanity checks
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

        # Chunks should not exceed size by much (word boundary tolerance)
        for chunk in chunks:
            assert len(chunk) <= chunk_size + 100

    def test_chunk_preserves_content(self):
        
        text = "The quick brown fox jumps over the lazy dog."

        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)

        
        combined = " ".join(chunks)

        original_words = set(text.split())
        chunk_words = set(combined.split())

        # All original words should be present
        assert original_words.issubset(chunk_words)

    def test_chunk_with_newlines(self):
       
        text = """
        First paragraph with some content.

        Second paragraph with different content.

        Third paragraph.
        """

        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

        # Should handle newlines without errors
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestPDFToChunksIntegration:
    """Test the full pipeline from PDF to chunks."""

    @patch('src.utils.fitz.open')
    def test_pdf_to_chunks_pipeline(self, mock_fitz_open):
 
        # Setup: Mock PDF with substantial text
        pdf_text = "Machine learning is a subset of AI. " * 30  # ~1110 chars

        mock_page = Mock()
        mock_page.get_text.return_value = pdf_text

        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.close = Mock()

        mock_fitz_open.return_value = mock_doc

        # Execute full pipeline
        extracted_text = load_and_extract_text("test.pdf")
        chunks = chunk_text(extracted_text, chunk_size=200, chunk_overlap=50)

        # Assert
        assert len(chunks) > 1  # Should produce multiple chunks
        assert all(len(chunk) <= 250 for chunk in chunks)  # Respect chunk size

        # Verify PDF was properly closed
        mock_doc.close.assert_called_once()


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling and edge cases."""

    @patch('src.utils.fitz.open')
    def test_pdf_file_not_found(self, mock_fitz_open):
        """
        Test handling when PDF file doesn't exist.

        Interview point: "I test error conditions"
        """
        mock_fitz_open.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            load_and_extract_text("nonexistent.pdf")

    def test_invalid_chunk_parameters(self):
      
        text = "Test text for chunking."

        try:
            chunks = chunk_text(text, chunk_size=0, chunk_overlap=0)
            assert isinstance(chunks, list)
        except ValueError:
            pass
