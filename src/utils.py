import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_extract_text(pdf_path: str) -> str:
    """
    Loads a PDF and extracts text from it.

    Args:
        pdf_path: The file path to the PDF.

    Returns:
        A single string containing all the text from the PDF.
    """
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits a long text into smaller, overlapping chunks.

    Args:
        text: The input text.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks