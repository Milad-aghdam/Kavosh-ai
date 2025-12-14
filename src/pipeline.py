import logging
import torch
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from src.config import get_openrouter_config
from src.confidence import ConfidenceScorer


logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.config = get_openrouter_config()
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.prompt = None
        self.pdf_name = None
        self.embeddings = None  # Store embeddings model for confidence calculation
        self.confidence_scorer = ConfidenceScorer()  # Initialize confidence scorer 

    def build(self, text_chunks):
        """
        Builds the Hybrid Search + Reranking pipeline.
        """

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing models on: {device_type.upper()}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device_type},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info("Building FAISS Vector Store...")
        self.vector_store = FAISS.from_texts(texts=text_chunks, embedding=self.embeddings)
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        logger.info("Building BM25 Index...")
        bm25_retriever = BM25Retriever.from_texts(text_chunks)
        bm25_retriever.k = 10

        logger.info("Initializing Hybrid Search (Ensemble)...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )


        logger.info("Initializing Reranker (BAAI/bge-reranker-v2-m3)...")
        
        reranker_model = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-v2-m3",
            model_kwargs={'device': device_type} 
        )

        compressor = CrossEncoderReranker(model=reranker_model, top_n=3)


        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
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
        logger.info("Pipeline built successfully with Hybrid Search + Reranking.")

    def query(self, question: str) -> dict:

        if not self.retriever or not self.llm or not self.prompt:
            return {
                "answer": "Pipeline not built.",
                "sources": [],
                "confidence": 0.0,
                "confidence_level": "unknown",
                "should_reject": True
            }

        logger.info(f"Querying: {question}")
        relevant_docs = self.retriever.invoke(question)


        reranker_confidence = self.confidence_scorer.calculate_reranker_confidence(
            relevant_docs
        )

        logger.info(f"Reranker confidence: {reranker_confidence}")

        if reranker_confidence["max_score"] < self.confidence_scorer.reranker_threshold:
            confidence_result = self.confidence_scorer.compute_overall_confidence(
                reranker_scores=reranker_confidence,
                answer_text=None
            )

            rejection_message = self.confidence_scorer.format_confidence_message(
                confidence_result
            )

            return {
                "answer": rejection_message,
                "sources": relevant_docs,  
                "confidence": confidence_result["confidence"],
                "confidence_level": confidence_result["confidence_level"],
                "should_reject": True,
                "confidence_details": confidence_result["details"]
            }

        context_for_prompt = "\n---\n".join([doc.page_content for doc in relevant_docs])
        formatted_prompt = self.prompt.format(context=context_for_prompt, question=question)
        response = self.llm.invoke(formatted_prompt)
        answer_text = response.content


        question_embedding = self.embeddings.embed_query(question)
        answer_embedding = self.embeddings.embed_query(answer_text)

        import numpy as np
        semantic_sim = self.confidence_scorer.calculate_semantic_similarity(
            np.array(question_embedding),
            np.array(answer_embedding)
        )


        confidence_result = self.confidence_scorer.compute_overall_confidence(
            reranker_scores=reranker_confidence,
            semantic_similarity=semantic_sim,
            answer_text=answer_text
        )

  
        if confidence_result["should_reject"]:

            final_answer = self.confidence_scorer.format_confidence_message(
                confidence_result
            )
        else:
            confidence_badge = self.confidence_scorer.format_confidence_message(
                confidence_result
            )
            final_answer = f"{confidence_badge}\n\n{answer_text}"

        return {
            "answer": final_answer,
            "sources": relevant_docs,
            "confidence": confidence_result["confidence"],
            "confidence_level": confidence_result["confidence_level"],
            "should_reject": confidence_result.get("should_reject", False),
            "confidence_details": confidence_result["details"]
        }