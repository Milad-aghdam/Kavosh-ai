import gradio as gr
from src.utils import load_and_extract_text, chunk_text
from src.pipeline import RAGPipeline
from src.translator import detect_language, translate_text 

pipeline = None

def process_pdf_and_query(pdf_file, question):
    global pipeline

    if pdf_file is None or not question.strip():
        return "Please upload a PDF and enter a question."

    original_lang = detect_language(question)
    print(f"Detected language: {original_lang}")
    
    question_for_pipeline = question
    if original_lang == 'fa':
        yield "Persian question detected. Translating to English for processing..."
        question_for_pipeline = translate_text(question, target_lang='en')
        print(f"Translated question: {question_for_pipeline}")

    if pipeline is None or not hasattr(pipeline, 'pdf_name') or pipeline.pdf_name != pdf_file.name:
        yield "Building knowledge base..."
        try:
            document_text = load_and_extract_text(pdf_file.name)
            text_chunks = chunk_text(document_text)
            pipeline = RAGPipeline()
            pipeline.build(text_chunks)
            pipeline.pdf_name = pdf_file.name
        except Exception as e:
            yield f"Error building pipeline: {e}"
            return

    yield "Searching for the answer..."
    try:
        result = pipeline.query(question_for_pipeline)
        answer_from_pipeline = result["answer"]
        sources = result["sources"]

        final_answer = answer_from_pipeline
        if original_lang == 'fa':
            yield "Translating answer back to Persian..."
            final_answer = translate_text(answer_from_pipeline, target_lang='fa')
        
        formatted_output = f"**پاسخ (Answer):**\n{final_answer}\n\n---\n\n**منابع (Sources):**\n"
        for i, doc in enumerate(sources):
            source_text = doc.page_content.replace('\n', ' ').strip()
            formatted_output += f"**[{i+1}]** {source_text[:250]}...\n\n"
            
        yield formatted_output

    except Exception as e:
        yield f"Error during query: {e}"

with gr.Blocks(theme=gr.themes.Soft(), title="Kavosh-AI") as demo:
    gr.Markdown("# 📚 Kavosh-AI: Your Personal Research Assistant (EN/FA)")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload your PDF / PDF خود را آپلود کنید")
            question_input = gr.Textbox(label="Ask a question / سوال خود را بپرسید")
            submit_button = gr.Button("Get Answer / دریافت پاسخ", variant="primary")

        with gr.Column(scale=2):
            answer_output = gr.Markdown(label="Answer & Sources / پاسخ و منابع")

    submit_button.click(
        fn=process_pdf_and_query,
        inputs=[pdf_input, question_input],
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)