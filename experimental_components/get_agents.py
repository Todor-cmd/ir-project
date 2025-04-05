from langchain_openai import ChatOpenAI
from .custom_retrievers import PineconeRetriever, HybridBM25Retriever, AutoMergeWithRerankRetriever
from .open_ai_agent import OpenAIAssistant
from .custom_agent import CustomAgent

def get_rag_agents_for_experiment_1():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    retriever_pinecone = PineconeRetriever()
    retriever_hybrid = HybridBM25Retriever(pinecone_preloaded=True)
    retriever_auto_merge = AutoMergeWithRerankRetriever()

    agent_pinecone = CustomAgent(llm, retriever_pinecone)
    agent_hybrid = CustomAgent(llm, retriever_hybrid)
    agent_auto_merge = CustomAgent(llm, retriever_auto_merge)
    agent_openai = OpenAIAssistant(llm)
    
    # It's important that pinecone is before hybrid in this dict if pinecone-preloaded is set to true!
    agents = {
        "openai": agent_openai,
        "pinecone": agent_pinecone,
        "hybrid": agent_hybrid,
        "auto_merge": agent_auto_merge
    }
    
    return agents

