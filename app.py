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
        confidence = result.get("confidence", 0.0)
        confidence_level = result.get("confidence_level", "unknown")
        should_reject = result.get("should_reject", False)
        confidence_details = result.get("confidence_details", {})

        # Log confidence information for monitoring
        logger.info(f"Confidence: {confidence:.3f} ({confidence_level})")
        logger.info(f"Confidence details: {confidence_details}")

        final_answer = answer_from_pipeline
        if original_lang == 'fa':
            yield "Translating answer back to Persian...", current_pipeline
            final_answer = translate_text(answer_from_pipeline, target_lang='fa')

        # Format output with confidence information
        formatted_output = f"**پاسخ (Answer):**\n{final_answer}\n\n"

        # Add confidence details section
        formatted_output += "---\n\n"
        formatted_output += f"**📊 Confidence Metrics:**\n"
        formatted_output += f"- Overall Confidence: {confidence:.2f} ({confidence_level})\n"
        formatted_output += f"- Retrieval Quality: {confidence_details.get('reranker_max', 0.0):.2f}\n"
        if confidence_details.get('semantic_similarity'):
            formatted_output += f"- Answer Relevance: {confidence_details['semantic_similarity']:.2f}\n"

        # Add sources
        formatted_output += "\n---\n\n**منابع (Sources):**\n"

        if should_reject:
            # For rejected answers, show what was found but explain why it wasn't good enough
            formatted_output += "*Note: The following passages were the closest matches found, but they may not directly answer your question.*\n\n"

        for i, doc in enumerate(sources):
            source_text = doc.page_content.replace('\n', ' ').strip()
            # Show reranker score for each source if available
            doc_score = doc.metadata.get('relevance_score', 'N/A')
            formatted_output += f"**[{i+1}]** (relevance: {doc_score if isinstance(doc_score, str) else f'{doc_score:.2f}'}) {source_text[:250]}...\n\n"

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
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False 
    )