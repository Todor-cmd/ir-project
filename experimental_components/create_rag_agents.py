from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

def create_openai_assistant(llm_model: str, vector_store_name: str):
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
    vector_store_id = os.getenv(f"OPENAI_VECTOR_STORE_ID_{vector_store_name.upper()}")
    
    assistant_id_env_key = f"OPENAI_ASSISTANT_ID_{vector_store_name.upper()}"
    
    # Check if assistant ID already exists in env
    existing_assistant_id = os.getenv(assistant_id_env_key)
    if existing_assistant_id:
        return client.beta.assistants.retrieve(existing_assistant_id)
    
    client = OpenAI()
    openai_assistant = client.beta.assistants.create(
        name="RAG Agent",
        instructions="You are a helpful assistant that can answer questions about the data in the vector store.",
        tools=[{"type": "vector_store", "vector_store_id": vector_store_id}],
        model=llm_model
    )
    # Store the assistant ID in the .env file
    with open('.env', 'a') as env_file:
        env_file.write(f'\n{assistant_id_env_key}={openai_assistant.id}')
    
    return openai_assistant
    

def get_custom_agent(llm: str, retriever: str, dataset_name: str):
    #TODO: Once we have decided on vector stores, embeddings and retrievers, we can create the custom agent using
    # the get_agent() method in the custom_agent.py file.
    pass
