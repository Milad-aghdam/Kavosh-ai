import logging
import torch 
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers.ensemble import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever
from src.config import get_openrouter_config


logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.config = get_openrouter_config()
        self.vector_store = None
        self.retriever = None 
        self.llm = None
        self.prompt = None
        self.pdf_name = None 

    def build(self, text_chunks):
        """
        Builds the Hybrid Search (Vector + Keyword) pipeline.
        """

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing BGE-M3 model on: {device_type.upper()}")


        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device_type},
            encode_kwargs={'normalize_embeddings': True} 
        )

        logger.info("Building FAISS Vector Store (Semantic Search)...")
        self.vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        logger.info("Building BM25 Index (Keyword Search)...")
        bm25_retriever = BM25Retriever.from_texts(text_chunks)
        bm25_retriever.k = 5

        logger.info("Initializing Hybrid Search (Ensemble)...")
        self.retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )

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
        logger.info("Pipeline built successfully with Hybrid Search.")

    def query(self, question: str) -> dict:
        """
        Queries the pipeline and returns a dictionary with the answer and sources.
        """
        if not self.retriever or not self.llm or not self.prompt:
            return {"answer": "Pipeline not built.", "sources": []}

        relevant_docs = self.retriever.invoke(question)
        
        context_for_prompt = "\n---\n".join([doc.page_content for doc in relevant_docs])
        
        formatted_prompt = self.prompt.format(context=context_for_prompt, question=question)
        response = self.llm.invoke(formatted_prompt)
        
        return {
            "answer": response.content,
            "sources": relevant_docs
        }