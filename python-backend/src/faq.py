import os
import json
import uuid
import time
import numpy as np
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import chromadb
from chromadb.utils import embedding_functions
import logging #
import yaml

def load_config_from_yaml(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
        return config if config is not None else {} # Return empty dict if YAML is empty

config_filepath = 'config/config.yaml'
app_config = load_config_from_yaml(config_filepath)

RAG_TOP_K = app_config.get('RAG_TOP_K')
RAG_CHUNK_OVERLAP = app_config.get('RAG_CHUNK_OVERLAP')
RAG_CHUNK_SIZE = app_config.get('RAG_CHUNK_SIZE')
LLM_TEMPERATURE = app_config.get('LLM_TEMPERATURE')
MY_MODEL = app_config.get('MY_MODEL')
KNOWLEDGE_BASE_DOCUMENT_PATH = app_config.get('KNOWLEDGE_BASE_DOCUMENT_PATH')
EMBEDDING_MODEL_NAME = app_config.get('EMBEDDING_MODEL_NAME')
RAG_COLLECTION_NAME = app_config.get('RAG_COLLECTION_NAME')
CHROMA_RAG_PATH = app_config.get('CHROMA_RAG_PATH')

logger = logging.getLogger(__name__) #

class TelcoKnowledgeBaseHandler:
    def __init__(
        self,
        document_path: str = KNOWLEDGE_BASE_DOCUMENT_PATH, #
        chroma_path: str = CHROMA_RAG_PATH, #
        collection_name: str = RAG_COLLECTION_NAME, #
        embedding_model_name: str = EMBEDDING_MODEL_NAME, #
        model: str = MY_MODEL, #
        llm_temperature: float = LLM_TEMPERATURE #
    ):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )

        self.collection_name = collection_name
        self.collection = self.chroma_client.get_or_create_collection(
                                name=self.collection_name,
                                embedding_function=self.embedding_function
                            )
        logger.info(f"ChromaDB initialized at: {chroma_path}")
        logger.info(f"Collection '{self.collection_name}' ready. Current items: {self.collection.count()}")

        # Add some initial products if the collection is empty
        if self.collection.count() == 0:
            logger.info("ChromaDB collection is empty. Adding initial example products...")
            docs = self.load_pdf(document_path)
            self.add_documents(
                doc_id="singtel_document",
                raw_documents=docs,
                source_metadata={"source_type": "internal_kb", "version": "1.0", "title": "Singtel Overview"}
            )

        self.llm = ChatOpenAI(
            temperature=llm_temperature,
            model_name=model,
            max_tokens=300,
        )

    def load_pdf(self, doc_path) -> List[Document]:
        loader = PyPDFLoader(doc_path)
        raw_docs = loader.load()
        logger.info(f"Loaded {len(raw_docs)} pages from PDF.")
        return raw_docs

    def split_documents(self, documents: List[Document], chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks.")
        return chunks

    def add_documents(self, doc_id: str, raw_documents: List[Document], source_metadata: Dict[str, Any]):
        chunks = self.split_documents(raw_documents)
        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i+1}"
            chunk_text = chunk.page_content
            metadata = {
                "document_id": doc_id,
                "chunk_index": i + 1,
                "chunk_length": len(chunk_text),
                **source_metadata,
                **chunk.metadata
            }
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk_text)
            chunk_metadatas.append(metadata)

        existing_ids = self.collection.get(ids=chunk_ids, include=[])['ids']
        new_ids = [cid for cid in chunk_ids if cid not in existing_ids]

        if not new_ids:
            logger.info(f"All chunks for document ID '{doc_id}' already exist. Skipping addition.") #
            return

        new_texts = [chunk_texts[i] for i, cid in enumerate(chunk_ids) if cid in new_ids]
        new_metadatas = [chunk_metadatas[i] for i, cid in enumerate(chunk_ids) if cid in new_ids]

        logger.info(f"Adding {len(new_texts)} new chunks to ChromaDB for document ID '{doc_id}'.") 
        self.collection.add(
            documents=new_texts,
            metadatas=new_metadatas,
            ids=new_ids
        )
        logger.info(f"Added {len(new_ids)} chunks to ChromaDB.")

    def query(self, query_text: str, top_k : int =RAG_TOP_K) -> List[str]:
        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        docs = results['documents'][0]
        logger.info(f"Query: {query_text}")
        logger.info(f"Retrieved {len(docs)} documents.")
        if len(docs) > 0:
            return docs
        else:
            return None
        
    def generate_answer(self, query: str, documents: List[str]) -> str:
        if not documents: #
            logger.info(f"No relevant documents found for query '{query}'. Returning default response.") #
            return "Sorry, I couldn't find an answer to your question based on my knowledge base. Can you please rephrase or ask something else?"

        context = "\n\n".join(documents)
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            ased on the following context, answer the question comprehensively and concisely.
            If the context does not contain enough information to answer the question,
            state that you cannot provide a definitive answer based on the provided information,
            and suggest rephrasing or asking a different question.
            Do not make up information.

            Context:
            {context}

            Question: {question}

            Answer:"""
        )
        response = self.llm.invoke(prompt.format(context=context, question=query))
        logger.info(f"Generated RAG answer: {response.content[:200]}...") #
        return response.content

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) #
    handler = TelcoKnowledgeBaseHandler()

    query = "When can I renew my telco contract plan?"
    retrieved_docs = handler.query(query)
    answer = handler.generate_answer(query, retrieved_docs)
    print(f"\nQuery: {query}\nAnswer: {answer}")
    print("-" * 50)

    query = "What is the capital of France?" # Irrelevant query
    retrieved_docs = handler.query(query)
    answer = handler.generate_answer(query, retrieved_docs)
    print(f"\nQuery: {query}\nAnswer: {answer}")
    print("-" * 50)

    query = "Tell me about Singtel's 5G coverage."
    retrieved_docs = handler.query(query)
    answer = handler.generate_answer(query, retrieved_docs)
    print(f"\nQuery: {query}\nAnswer: {answer}")