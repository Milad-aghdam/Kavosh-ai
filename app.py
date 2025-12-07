import gradio as gr
from src.utils import load_and_extract_text, chunk_text
from src.pipeline import RAGPipeline
from src.translator import detect_language, translate_text 

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_pdf_and_query(pdf_file, question, pipeline_state):
    """
    Args:
        pdf_file: The file object uploaded by the user.
        question: The text string question.
        pipeline_state: The PRIVATE notebook for this specific user.
    """

    if pdf_file is None or not question.strip():
        yield "Please upload a PDF and enter a question.", pipeline_state
        return

    original_lang = detect_language(question)
    logger.info(f"Detected language: {original_lang}") 
    
    question_for_pipeline = question
    
    if original_lang == 'fa':
        yield "Persian question detected. Translating...", pipeline_state 
        question_for_pipeline = translate_text(question, target_lang='en')
        logger.info(f"Translated question: {question_for_pipeline}")

    current_pipeline = pipeline_state

    if current_pipeline is None or not hasattr(current_pipeline, 'pdf_name') or current_pipeline.pdf_name != pdf_file.name:
        
        yield "Building knowledge base (this takes a moment)...", current_pipeline
        
        try:
            logger.info(f"Loading PDF: {pdf_file.name}")
            document_text = load_and_extract_text(pdf_file.name)
            text_chunks = chunk_text(document_text)
            
            new_pipeline = RAGPipeline()
            new_pipeline.build(text_chunks)
            new_pipeline.pdf_name = pdf_file.name
            
            current_pipeline = new_pipeline
            logger.info("Pipeline built successfully.")
            
        except Exception as e:
            logger.error(f"Error building pipeline: {e}")
            yield f"Error building pipeline: {e}", current_pipeline
            return

    yield "Searching for the answer...", current_pipeline
    
    try:
        result = current_pipeline.query(question_for_pipeline)
        answer_from_pipeline = result["answer"]
        sources = result["sources"]

        final_answer = answer_from_pipeline
        if original_lang == 'fa':
            yield "Translating answer back to Persian...", current_pipeline
            final_answer = translate_text(answer_from_pipeline, target_lang='fa')
        
        formatted_output = f"**پاسخ (Answer):**\n{final_answer}\n\n---\n\n**منابع (Sources):**\n"
        for i, doc in enumerate(sources):
            source_text = doc.page_content.replace('\n', ' ').strip()
            formatted_output += f"**[{i+1}]** {source_text[:250]}...\n\n"
            
        logger.info("Query successful.")
        
        yield formatted_output, current_pipeline

    except Exception as e:
        logger.error(f"Error during query: {e}")
        yield f"Error during query: {e}", current_pipeline

with gr.Blocks(theme=gr.themes.Soft(), title="Kavosh-AI") as demo:
    gr.Markdown("# 📚 Kavosh-AI: Your Personal Research Assistant (EN/FA)")
    session_pipeline = gr.State(value=None) 

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload your PDF / PDF خود را آپلود کنید")
            question_input = gr.Textbox(label="Ask a question / سوال خود را بپرسید")
            submit_button = gr.Button("Get Answer / دریافت پاسخ", variant="primary")

        with gr.Column(scale=2):
            answer_output = gr.Markdown(label="Answer & Sources / پاسخ و منابع")

    submit_button.click(
        fn=process_pdf_and_query,
        inputs=[pdf_input, question_input, session_pipeline],
        
        outputs=[answer_output, session_pipeline]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)