import os
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore

from typing import List, Optional
from pydantic import Field
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.openai import OpenAIEmbedding

from abc import ABC, abstractmethod
from llama_index.core.schema import MetadataMode
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.llms.groq import Groq
from llama_index.core import QueryBundle
from llama_index.core.postprocessor import LLMRerank
from pprint import pprint
import time

class CustomRetriever(ABC):
    @abstractmethod
    def add_documents(self, documents) -> None:
        pass
    
    @abstractmethod
    def retrieve_relevant_documents(self, query: str, k: int) -> List[str]:
        pass

class PineconeRetriever(CustomRetriever):
    def __init__(self):
        """Initialize the PineconeRetriever with Pinecone credentials."""
        super().__init__()

        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pc_index = self.pc.Index(host=os.getenv("PINECONE_HOST"))
        
        self.splitter = SentenceSplitter(chunk_size=500, chunk_overlap=100)
        
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
        
    def add_documents(self, documents : List[Document]) -> None:
        """Add documents to the Pinecone index.
        
        Args:
            documents: List of Document objects to add to the index
        """
        # First delete all documents from the index
        self.delete_all_documents()
        
        # Split documents into chunks
        nodes = self.splitter.get_nodes_from_documents(documents)
       
        # Add documents to Pinecone
        self.docstore = SimpleDocumentStore()
        self.docstore.add_documents(nodes)
        
        self.storage_context = StorageContext.from_defaults(
                docstore=self.docstore, vector_store=self.vector_store
            )
            
        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            vector_store=self.vector_store
        )
        
        # Needed for vectors to sync
        print("Sleeping for 60 seconds")
        time.sleep(60.0)

        stats = self.pc_index.describe_index_stats()
        
        print(f"Added {len(documents)} documents ({len(nodes)} chunks) to Pinecone index resulting in {stats.total_vector_count} vectors")
    
    def retrieve_relevant_documents(self, query: str, k: int = 5):
        """Retrieve relevant documents given a query.
        
        Args:
            query: The query string
            k: Number of documents to retrieve (default: 5)
            
        Returns:
            List of retrieved Node objects containing the relevant documents
        """
  
        
        # Create retriever with specified parameters
        retriever = self.index.as_retriever(similarity_top_k=k)
        # Retrieve relevant documents
        retrieved_nodes = retriever.retrieve(query)
        
        # Print the actual texts for debugging
        retrieved_texts = [node.text for node in retrieved_nodes]
            
        return retrieved_texts
    
# TODO: Use QueryFusionRetriever for this
class HybridBM25Retriever(CustomRetriever):
    def __init__(self, pinecone_preloaded=True):
        super().__init__()
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pc_index = self.pc.Index(host=os.getenv("PINECONE_HOST"))
        
        # Setup vector store and storage context
        self.vector_store = PineconeVectorStore(
            index=self.pc_index,
            index_name=os.getenv("PINECONE_INDEX_NAME")
        )
        
        self.pinecone_preloaded = pinecone_preloaded
        self.index = None
        self.splitter = SentenceSplitter(chunk_size=500, chunk_overlap=100)


    def _refresh_pinecone(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pc_index = self.pc.Index(host=os.getenv("PINECONE_HOST"))
        
        # Setup vector store and storage context
        self.vector_store = PineconeVectorStore(
            index=self.pc_index,
            index_name=os.getenv("PINECONE_INDEX_NAME")
        )
    
    def delete_all_documents(self) -> None:
        """Delete all documents from the Pinecone vector store."""
        # Check if index has any vectors
        stats = self.pc_index.describe_index_stats()
        if stats.total_vector_count > 0:
            self.vector_store.clear()
            print(f"Cleared {stats.total_vector_count} vectors from index")
            stats = self.pc_index.describe_index_stats()
            print(stats.total_vector_count)
        else:
            print("Index is already empty")
        
        self.nodes = []
    
    
    def add_documents(self, documents : List[Document]):
        nodes = self.splitter.get_nodes_from_documents(documents)
        self.docstore = SimpleDocumentStore()
        self.docstore.add_documents(nodes)
        if self.pinecone_preloaded:
            #Load data directly from Pinecone
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            stats = self.pc_index.describe_index_stats()
            print(stats.total_vector_count)
            
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context
            )
            
        else:
            self.delete_all_documents()            
            
            self.storage_context = StorageContext.from_defaults(
                    docstore=self.docstore, vector_store=self.vector_store
                )
                
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                vector_store=self.vector_store
            )
            
            # Needed for vectors to sync
            print("Sleeping for 60 seconds")
            time.sleep(60.0)

            stats = self.pc_index.describe_index_stats()
            print(stats.total_vector_count)
            
            print(f"Added {len(documents)} documents ({len(nodes)} chunks) to Pinecone index")

    def retrieve_relevant_documents(self, query, k=5):
        """Retrieve documents using BM25 and dense retrieval, then merge scores."""
        
        retriever = QueryFusionRetriever(
            [
                self.index.as_retriever(similarity_top_k=k),
                BM25Retriever.from_defaults(
                    docstore=self.docstore, similarity_top_k=k
                ),
            ],
            num_queries=1,
            use_async=True,
        )
        
        # Retrieve documents
        retrieved_nodes = retriever.retrieve(query)
        retrieved_dict = {doc.node.text: doc.score for doc in retrieved_nodes}
        sorted_results = sorted(retrieved_dict.items(), key=lambda x: x[1], reverse=True)
        # return [doc for doc, _ in sorted_results]
        return [node.text for node in retrieved_nodes]
 

class AutoMergeWithRerankRetriever(CustomRetriever):
    def __init__(self):
        super().__init__()
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        self.llm = Groq(model="llama-3.3-70b-versatile")
        self.retriever = None
        


        
    def add_documents(self, documents):
        # Ensure documents are of type Document
        doc_objects = [
            doc if isinstance(doc, Document) else Document(text=doc)
            for doc in documents
        ]
        
        # Combine all documents into one
        documents = [
            Document(text="\n\n".join(
                    document.get_content(metadata_mode=MetadataMode.ALL) 
                    for document in doc_objects
                )
            )
        ]
        
        node_parser = HierarchicalNodeParser.from_defaults()
        nodes = node_parser.get_nodes_from_documents(documents)
        
        leaf_nodes = get_leaf_nodes(nodes)
        root_nodes = get_root_nodes(nodes)
        
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)


        storage_context = StorageContext.from_defaults(docstore=docstore)

        base_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )
        base_index.storage_context.persist(persist_dir="./data/custom_retriever_storage")
        
        base_retriever = base_index.as_retriever(similarity_top_k=20)
        self.retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)
    
    def retrieve_relevant_documents(self, query, k=5):
        query_bundle = QueryBundle(query_str=query)
        auto_merged_nodes = self.retriever.retrieve(query_bundle)
        reranker = LLMRerank(
            llm=self.llm,
            choice_batch_size=5,
            top_n=k,
        )
        reranked_nodes = reranker.postprocess_nodes(auto_merged_nodes, query_bundle=query_bundle)
        return [node.text for node in reranked_nodes]
        

if __name__ == "__main__":
    from llama_index.readers.file import PyMuPDFReader
    
    retriever = HybridBM25Retriever(pinecone_preloaded=False)
    reader = PyMuPDFReader()
    
    # # Load all PDF documents from the data directory
    pdf_dir = "./data/sse_lectures"
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    pdf_files = pdf_files[:3]
    documents = []
    for pdf_file in pdf_files:
        docs = reader.load(file_path=pdf_file)
        # join all the documents into one
        doc_text = "\n".join([doc.text for doc in docs])
        documents.append(Document(text=doc_text))
        print(f"Loaded document from {pdf_file}")
    
    # Add documents to retriever
    retriever.add_documents(documents)
    
    query = "What is sustainable software engineering?"
    docs = retriever.retrieve_relevant_documents(query, k=3)
    pprint(docs)
    
    # retriever.delete_all_documents()
    
    
    
