import os
from openai import OpenAI

def get_openai_assistant(llm_model: str):
    """
    Creates and returns an OpenAI assistant configured for RAG (Retrieval Augmented Generation).

    This function initializes an OpenAI assistant that can access a vector store to answer questions.
    The assistant uses the specified LLM model and is configured with instructions to be helpful
    and answer questions based on data in the connected vector store.

    Args:
        llm_model (str): The identifier of the OpenAI model to use (e.g. "gpt-4", "gpt-3.5-turbo")

    Returns:
        Assistant: An OpenAI Assistant object configured with vector store access
    """
    # This variable needs to be set. Its automatically done by running 'verctorize_data.py'
    vector_store_id = os.getenv("OPENAI_VECTOR_STORE_ID")
    client = OpenAI()
    openai_assistant = client.beta.assistants.create(
        name="RAG Agent",
        instructions="You are a helpful assistant that can answer questions about the data in the vector store.",
        tools=[{"type": "vector_store", "vector_store_id": vector_store_id}],
        model=llm_model
    )
    return openai_assistant
    

def get_custom_agent(llm: str, retriever: str):
    #TODO: Once we have decided on vector stores, embeddings and retrievers, we can create the custom agent using
    # the get_agent() method in the custom_agent.py file.
    pass
