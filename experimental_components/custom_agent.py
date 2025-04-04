from langchain_core.language_models.llms import BaseLLM
from .custom_retrievers import CustomRetriever


class CustomAgent:
    def __init__(self, llm : BaseLLM, retriever : CustomRetriever) -> None:
        self.llm = llm
        self.retriever = retriever
    
    def load_documents(self, documents):
        """Load documents into the retriever if supported"""
        self.retriever.add_documents(documents)
        
    
    def get_most_relevant_docs(self, query, k=3):
        """Retrieve the most relevant documents for the query"""
        return self.retriever.retrieve_relevant_documents(query, k)
        
    def generate_answer(self, query, relevant_docs):
        """Generate an answer using the LLM based on relevant documents"""
        context = "\n\n---Source SEPARATOR---\n\n".join([doc for doc in relevant_docs])
        
        system_prompt = f"""You are a helpful AI assistant. Use the following sources to answer the user's question:
        
        Sources: {context}
        
        Answer the question based on the sources provided. If you cannot find the answer in the sources, say so, but 
        also try to answer the question anyway if you can. Keep the answer concise and to the point."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = self.llm.invoke(messages)
        return response.content

