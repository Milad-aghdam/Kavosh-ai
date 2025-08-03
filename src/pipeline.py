%%writefile src/pipeline.py

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from src.config import get_openrouter_config

class RAGPipeline:
    def __init__(self):
        self.config = get_openrouter_config()
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.prompt = None

    def build(self, text_chunks):
        """Builds the entire RAG pipeline."""
        print("Initializing free embedding model (HuggingFace)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Building vector store with free embeddings...")
        self.vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        print("Vector store built successfully.")
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        template = """
        You are an academic research assistant. Use the following pieces of context from a document to answer the question at the end.
        If you don't know the answer, just say that you don't know.
        Provide a concise and accurate answer based only on the provided context.

        Context: {context}
        Question: {question}
        Helpful Answer:
        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        self.llm = ChatOpenAI(
            model="mistralai/mistral-7b-instruct:free",
            openai_api_key=self.config["api_key"],
            openai_api_base=self.config["base_url"],   
            temperature=0
        )

    def query(self, question: str) -> str:
        """Queries the pipeline with a question and returns the LLM's answer."""
        if not self.retriever or not self.llm or not self.prompt:
            return "Pipeline not built. Please call the 'build' method first."
        
        relevant_docs = self.retriever.invoke(question)
        formatted_prompt = self.prompt.format(context=relevant_docs, question=question)
        response = self.llm.invoke(formatted_prompt)

        return response.content