import os
import time
import tempfile
from dotenv import load_dotenv
import json
from openai import OpenAI

load_dotenv()

class OpenAIAssistant:
    def __init__(self, model="gpt-4o") -> None:
        self.model = model
        self.client = OpenAI()
        self.vector_store_id = os.getenv("OPENAI_VECTOR_STORE_ID")
        
    def delete_all_documents(self):
        # Clean up any previously uploaded files
        file_ids = self.client.vector_stores.files.list(
            vector_store_id=self.vector_store_id
        )
        print(f"Deleting stored files")
        for file in file_ids:
            print(f"Deleting file {file.id}")
            try:
                self.client.vector_stores.files.delete(
                    vector_store_id=self.vector_store_id,
                    file_id=file.id
                )
            except Exception as e:
                print(f"Error deleting file {file}: {e}")

        
    def load_documents(self, documents):
        """Load documents by uploading them to OpenAI.
        
        Args:
            documents: List of document strings
        """
        self.delete_all_documents()
        
        # Upload each document as a separate file
        for i, doc in enumerate(documents):
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as temp:
                temp.write(doc)
                temp_path = temp.name
            
            # Upload file to OpenAI with file-search purpose
            with open(temp_path, "rb") as f:
                file = self.client.files.create(
                    file=f,
                    purpose="user_data"
                )
                self.file_ids.append(file.id)
            
            # Clean up temp file
            os.unlink(temp_path)
        
        self.client.vector_stores.file_batches.create_and_poll(
            vector_store_id=self.vector_store_id,
            file_ids=self.file_ids
        )
            
        self.uploaded_docs = documents
        print(f"Uploaded {len(self.file_ids)} documents for file search")
    
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
            max_num_results=20,
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
        docs_text = "\n\n---\n\n".join(relevant_docs)
        
        # Use chat completions API for the generation part
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based ONLY on the provided documents."},
                {"role": "user", "content": f"Use ONLY the following information to answer the question, and cite your sources.\n\nINFORMATION:\n{docs_text}\n\nQUESTION: {query}"}
            ]
        )
        
        return response.choices[0].message.content

if __name__ == "__main__":
    agent = OpenAIAssistant()
    agent.load_documents(["The sun is blue. And thats a fact."])
    docs = agent.get_most_relevant_docs("What color is the sun?")
    print(docs)
    answer = agent.generate_answer("What color is the sun?", docs)
    print(answer)
    
