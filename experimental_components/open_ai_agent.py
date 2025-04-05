import os
import time
import tempfile
from dotenv import load_dotenv
import json
from openai import OpenAI
from llama_index.core import Document
from typing import List
from langchain_openai import ChatOpenAI
from tqdm import tqdm
load_dotenv(override=True)

class OpenAIAssistant:
    def __init__(self, llm : ChatOpenAI) -> None:
        self.client = OpenAI()
        self.vector_store_id = os.getenv("OPENAI_VECTOR_STORE_ID")
        print(f"Vector store ID: {self.vector_store_id}")
        self.llm = llm
        
    def delete_all_files(self):
        file_ids = self.client.files.list()
        
        print(f"Deleting {len(file_ids.data)} files")
        for file in tqdm(file_ids.data):
            self.client.files.delete(file_id=file.id)
        
    def delete_all_documents(self):
        # Clean up any previously uploaded files
        file_ids = self.client.vector_stores.files.list(
            vector_store_id=self.vector_store_id
        )
        print(f"Deleting files from vector store {self.vector_store_id}")
        for file in tqdm(file_ids):
            try:
                self.client.vector_stores.files.delete(
                    vector_store_id=self.vector_store_id,
                    file_id=file.id
                )
            except Exception as e:
                print(f"Error deleting file {file}: {e}")

        
    def load_documents(self, documents: List[Document]):
        """Load documents by uploading them to OpenAI.
        
        Args:
            documents: List of document strings
        """
        self.delete_all_documents()
        self.delete_all_files()
        
        file_ids = []
        # Upload each document as a separate file
        print(f"Creating {len(documents)} temporary files")
        for i, doc in tqdm(enumerate(documents)):
            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".txt", delete=False) as temp:
                temp.write(doc.text)
                temp_path = temp.name
            
            # Upload file to OpenAI with file-search purpose
            with open(temp_path, "rb") as f:
                file = self.client.files.create(
                    file=f,
                    purpose="user_data"
                )
                file_ids.append(file.id)
            
            # Clean up temp file
            os.unlink(temp_path)
        
        print(f"Uploading {len(file_ids)} documents to OpenAI for file search")
        self.client.vector_stores.file_batches.create_and_poll(
            vector_store_id=self.vector_store_id,
            file_ids=file_ids
        )
            
        self.uploaded_docs = documents
        print(f"Uploaded {len(file_ids)} documents for file search")
    
    def get_most_relevant_docs(self, query):
        """Retrieve the most relevant documents for a query using file_search.
        
        Args:
            query: The query string
            
        Returns:
            List of relevant document strings
        """
        
        response = self.client.vector_stores.search(
            query=query,
            vector_store_id=self.vector_store_id,
            max_num_results=3,
            rewrite_query=True
        )
        
        retrieved_docs = []
        for file in response.data:
            for content_item in file.content:
                # Access attributes directly since we're working with Pydantic models
                if content_item.type == "text":
                    retrieved_docs.append(content_item.text)
        
        return retrieved_docs
        
    def generate_answer(self, query, relevant_docs):
        """Generate an answer for a query based on the relevant documents.
        
        Args:
            query: The query string
            relevant_docs: List of relevant document strings
            
        Returns:
            Generated answer string
        """
        # Combine the relevant docs into a prompt
        context = "\n\n---Source SEPARATOR---\n\n".join([doc for doc in relevant_docs])
        
        system_prompt = f"""You are a helpful AI assistant. Use the following sources to answer the user's question:
        
        Sources: {context}
        
        Answer the question based on the sources provided. If you cannot find the answer in the sources, say so, but 
        also try to answer the question anyway if you can. Keep the answer concise and to the point."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Use chat completions API for the generation part
        response = self.llm.invoke(messages)
        
        return response.content

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = OpenAIAssistant(llm)
    # agent.delete_all_files()
    # Can test the functions here if you want
    
