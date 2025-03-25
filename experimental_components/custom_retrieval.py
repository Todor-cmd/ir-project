from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from langchain.schema import BaseRetriever
from typing import List, Optional
from llama_index.core import Document
from pydantic import Field

# Use llama_index to implement a custom retrieval agent.
class CustomBM25Retriever(BaseRetriever):
    documents: List[Document] = Field(default_factory=list)
    nodes: Optional[List] = Field(default_factory=list)
    retriever: Optional[BM25Retriever] = None
    
    def __init__(self, documents=[]):
        super().__init__()
        self.documents = documents
        self.nodes = None
        self.retriever = None
        
    def add_documents(self, documents):
        self.documents = documents
        splitter = SentenceSplitter(chunk_size=512)
        parsed_docs = [Document(text=doc) for doc in documents]
        print(parsed_docs[0])
        self.nodes = splitter.get_nodes_from_documents(parsed_docs)
    
    def get_relevant_documents(self, query):
        if len(self.documents) == 0:
            raise ValueError("No documents loaded")
        if not self.retriever:
            self.retriever = BM25Retriever.from_defaults(nodes=self.nodes, similarity_top_k=5)
            
        retrieved_nodes = self.retriever.retrieve(query)
        return retrieved_nodes
    