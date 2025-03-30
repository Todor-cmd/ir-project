import os
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from pinecone import ServerlessSpec
from pinecone import Pinecone
from pinecone import ServerlessSpec

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever

from typing import List, Optional
from pydantic import Field
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np

class PineconeRetriever:
    def __init__(self):
        """Initialize the PineconeRetriever with Pinecone credentials."""

        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pc_index = self.pc.Index(host=os.getenv("PINECONE_HOST"))
        
        # Setup vector store and storage context
        self.vector_store = PineconeVectorStore(
            index=self.pc_index,
            index_name=os.getenv("PINECONE_INDEX_NAME")
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Create index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context
        )
        
    def delete_all_documents(self) -> None:
        """Delete all documents from the Pinecone vector store."""
        # Check if index has any vectors
        stats = self.pc_index.describe_index_stats()
        if stats.total_vector_count > 0:
            self.vector_store.clear()
            print(f"Cleared {stats.total_vector_count} vectors from index")
        else:
            print("Index is already empty")
        
    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the Pinecone index.
        
        Args:
            documents: List of Document objects to add to the index
        """
        # Convert strings to Document objects if needed
        doc_objects = [
            doc if isinstance(doc, Document) else Document(text=doc)
            for doc in documents
        ]
        
        # Add documents to existing index
        self.index.refresh_ref_docs(doc_objects)
        
        print(f"Added {len(documents)} documents to Pinecone index")
    
    def retrieve_relevant_documents(self, query: str, k: int = 5):
        """Retrieve relevant documents given a query.
        
        Args:
            query: The query string
            k: Number of documents to retrieve (default: 5)
            
        Returns:
            List of retrieved Node objects containing the relevant documents
        """
        # Create retriever with specified parameters
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=k,
        )
        # Retrieve relevant documents
        retrieved_nodes = retriever.retrieve(query)
        
        # Print the actual texts for debugging
        retrieved_texts = [node.text for node in retrieved_nodes]
            
        return retrieved_texts
    
# TODO: Use QueryFusionRetriever for this
class HybridBM25Retriever:
    documents: List[Document] = Field(default_factory=list)
    nodes: Optional[List] = Field(default_factory=list)
    bm25_retriever: Optional[BM25Retriever] = None
    dense_retriever: Optional[VectorIndexRetriever] = None
    vector_store: Optional[PineconeVectorStore] = None
    embedding_model: Optional[OpenAIEmbedding] = None

    def __init__(self, documents=[]):
        super().__init__()
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pc_index = self.pc.Index(host=os.getenv("PINECONE_HOST"))
        
        # Setup vector store and storage context
        self.vector_store = PineconeVectorStore(
            index=self.pc_index,
            index_name=os.getenv("PINECONE_INDEX_NAME")
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Create index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context
        )
        self.documents = documents
        self.nodes = None
        self.bm25_retriever = None
        self.dense_retriever = None
        
    
    def delete_all_documents(self) -> None:
        """Delete all documents from the Pinecone vector store."""
        # Check if index has any vectors
        stats = self.pc_index.describe_index_stats()
        if stats.total_vector_count > 0:
            self.vector_store.clear()
            print(f"Cleared {stats.total_vector_count} vectors from index")
        else:
            print("Index is already empty")
        self.documents = []
        self.nodes = []
    
    
    def add_documents(self, documents):
        doc_objects = [
            doc if isinstance(doc, Document) else Document(text=doc)
            for doc in documents
        ]
        self.documents = doc_objects
        
        self.index.refresh_ref_docs(doc_objects)
        
        print(f"Added {len(documents)} documents to Pinecone index")
        splitter = SentenceSplitter(chunk_size=512)
        parsed_docs = [Document(text=doc) for doc in documents]
        self.nodes = splitter.get_nodes_from_documents(parsed_docs)
    
    def get_relevant_documents(self, query, k=5, alpha=0.5):
        """Retrieve documents using BM25 and dense retrieval, then merge scores."""
        
        # BM25 Retriever
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=self.nodes, similarity_top_k=k)

        # Dense Retriever
        self.dense_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=k,
        )
        
        if len(self.documents) == 0:
            raise ValueError("No documents loaded")

        # Retrieve from BM25
        bm25_results = self.bm25_retriever.retrieve(query)
        bm25_dict = {doc.node.text: doc.score for doc in bm25_results}

        # Retrieve from Dense Retriever
        dense_results = self.dense_retriever.retrieve(query)
        dense_dict = {doc.node.text: doc.score for doc in dense_results}

        # Merge results with weighted sum
        merged_scores = {}
        for text in set(bm25_dict.keys()).union(dense_dict.keys()):
            bm25_score = bm25_dict.get(text, 0)
            dense_score = dense_dict.get(text, 0)
            merged_scores[text] = alpha * bm25_score + (1 - alpha) * dense_score

        # Sort results by merged score
        sorted_results = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in sorted_results]
 
    

    
if __name__ == "__main__":
    retriever = PineconeRetriever()
    retriever.delete_all_documents()
    retriever.add_documents(["The sky is blue", "The sky is sometimes yellow.", "Hulk smash"])
    docs = retriever.retrieve_relevant_documents("What colour is the sky?", k=2)
    print(docs)
