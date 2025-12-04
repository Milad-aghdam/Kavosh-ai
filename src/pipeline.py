from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from src.config import get_openrouter_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.config = get_openrouter_config()
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.prompt = None

    def build(self, text_chunks):
        logger.info("Initializing BGE-M3 embedding model (this may take a moment to download)...")
        embeddings = HuggingFaceEmbeddings(
          model_name="BAAI/bge-m3",
          model_kwargs={'device': 'cpu'},
          encode_kwargs={'normalize_embeddings': True}
      )
        print("Building vector store with free embeddings...")
        self.vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        template = """
        You are an academic research assistant. Use the following pieces of context to answer the question at the end.
        Your answer must be based *only* on the provided context.
        After providing the answer, list the exact context passages you used under a 'SOURCES:' heading.
        Do not make up any information. If you don't know the answer from the context, say so.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        self.llm = ChatOpenAI(
            model="mistralai/mistral-7b-instruct:free",
            openai_api_key=self.config["api_key"],
            openai_api_base=self.config["base_url"],
            temperature=0
        )
        print("Pipeline built successfully.")

    def query(self, question: str) -> dict:
        """
        Queries the pipeline and returns a dictionary with the answer and sources.
        """
        if not self.retriever or not self.llm or not self.prompt:
            return {"answer": "Pipeline not built. Please call 'build' method first.", "sources": []}

        relevant_docs = self.retriever.invoke(question)
        context_for_prompt = "\n---\n".join([doc.page_content for doc in relevant_docs])
        formatted_prompt = self.prompt.format(context=context_for_prompt, question=question)
        response = self.llm.invoke(formatted_prompt)
        return {
            "answer": response.content,
            "sources": relevant_docs
        }