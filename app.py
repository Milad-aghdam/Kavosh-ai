import gradio as gr
import os
from src.utils import load_and_extract_text, chunk_text
from src.pipeline import RAGPipeline

def process_pdf_and_query(pdf_file, question):
    """
    Takes a PDF file and a question, builds the RAG pipeline,
    and returns the answer.
    """
    if pdf_file is None or not question:
        return "Please upload a PDF and enter a question."

    pdf_path = pdf_file.name
    print(f"Processing PDF: {pdf_path}")
    print(f"Received question: {question}")

    try:
        yield "Step 1/3: Loading and chunking the document..."
        document_text = load_and_extract_text(pdf_path)
        text_chunks = chunk_text(document_text)

        yield "Step 2/3: Building the knowledge base..."
        pipeline = RAGPipeline()
        pipeline.build(text_chunks)
        yield "Step 3/3: Searching for the answer..."
        answer = pipeline.query(question)

        yield answer

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred during processing: {e}"


with gr.Blocks(theme=gr.themes.Soft(), title="Kavosh-AI") as demo:
    gr.Markdown(
        """
        # 📚 Kavosh-AI: Your Personal Research Assistant
        Upload a PDF document, ask a question, and get a precise, cited answer from the text.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload your PDF")
            question_input = gr.Textbox(label="Ask a question about the document")
            submit_button = gr.Button("Get Answer", variant="primary")

        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="Answer & Status", interactive=False, lines=15)

    submit_button.click(
        fn=process_pdf_and_query,
        inputs=[pdf_input, question_input],
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)