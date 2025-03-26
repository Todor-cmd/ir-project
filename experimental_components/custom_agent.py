from langchain_core.language_models.llms import BaseLLM
from langchain_core.retrievers import BaseRetriever

class CustomAgent:
    def __init__(self, llm : BaseLLM, retriever : BaseRetriever) -> None:
        self.llm = llm
        self.retriever = retriever
    
    def load_documents(self, documents):
        """Load documents into the retriever if supported"""
        if hasattr(self.retriever, 'add_documents'):
            self.retriever.add_documents(documents)
        else:
            raise NotImplementedError("This retriever doesn't support adding documents")
    
    def get_most_relevant_docs(self, query):
        """Retrieve the most relevant documents for the query"""
        return self.retriever.get_relevant_documents(query)
        
    def generate_answer(self, query, relevant_docs):
        """Generate an answer using the LLM based on relevant documents"""
        context = "\n\n".join([doc for doc in relevant_docs])
        
        system_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question:
        
        Context: {context}
        
        Answer the question based on the context provided. If you cannot find the answer in the context, say so."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = self.llm.invoke(messages)
        return response.content
