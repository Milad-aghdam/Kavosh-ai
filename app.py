import gradio as gr
from src.utils import load_and_extract_text, chunk_text
from src.pipeline import RAGPipeline

pipeline = None

def process_pdf_and_query(pdf_file, question):
    global pipeline
    if pipeline is None or not hasattr(pipeline, 'pdf_name') or pipeline.pdf_name != pdf_file.name:
        yield "New PDF detected. Building knowledge base..."
        try:
            pdf_path = pdf_file.name
            
            yield "Step 1/3: Loading and chunking the document..."
            document_text = load_and_extract_text(pdf_path)
            text_chunks = chunk_text(document_text)

            yield "Step 2/3: Building the knowledge base (This may take a moment)..."
            pipeline = RAGPipeline()
            pipeline.build(text_chunks)
            pipeline.pdf_name = pdf_file.name 

            yield "Step 3/3: Knowledge base built. Ready to answer questions."
        except Exception as e:
            yield f"Error building pipeline: {e}"
            return

    if not question:
        yield "Please enter a question to get an answer."
        return

    yield "Searching for the answer..."
    try:
        result = pipeline.query(question)
        
        answer = result["answer"]
        sources = result["sources"]
        
        formatted_output = f"**Answer:**\n{answer}\n\n---\n\n**Sources:**\n"
        for i, doc in enumerate(sources):
            source_text = doc.page_content.replace('\n', ' ').strip()
            formatted_output += f"**[{i+1}]** {source_text[:250]}...\n\n"
            
        yield formatted_output

    except Exception as e:
        yield f"Error during query: {e}"


with gr.Blocks(theme=gr.themes.Soft(), title="Kavosh-AI") as demo:
    gr.Markdown("# 📚 Kavosh-AI: Your Personal Research Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload your PDF")
            question_input = gr.Textbox(label="Ask a question about the document")
            submit_button = gr.Button("Get Answer", variant="primary")

        with gr.Column(scale=2):
            answer_output = gr.Markdown(label="Answer & Sources")

    submit_button.click(
        fn=process_pdf_and_query,
        inputs=[pdf_input, question_input],
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)