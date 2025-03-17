from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import time
import os


def vectorize_to_openai_vector_store(data_path: str, name: str):
        """
        Vectorizes data from a file and stores it in an OpenAI vector store.

        This function:
        1. Creates a file in OpenAI's system from the provided data path
        2. Creates a new vector store with the given name
        3. Adds the file to the vector store and waits for processing to complete
        4. Stores the vector store ID in a .env file for later use

        Args:
            data_path (str): Path to the data file to be vectorized
            name (str): Name to give the vector store

        Returns:
            None

        Raises:
            Exception: If the file fails to be added to the vector store
        """
        client = OpenAI()
        file = client.files.create(
            file=open(data_path, "rb"),
            purpose="assistants"
        )
        vector_store = client.beta.vector_stores.create(
            name=name
        )
        
        vector_store_file = client.beta.vector_stores.files.create_and_poll(
            vector_store_id=vector_store.id,
            file_id=file.id,
        )
        
        # Confirm the file was added
        while vector_store_file.status == "in_progress":
            time.sleep(1)
        if vector_store_file.status == "completed":
            print("File added to vector store")
        elif vector_store_file.status == "failed":
            raise Exception("Failed to add file to vector store")
        
        print (vector_store.id)
       
        # Set the vector store in the .env file
        vector_store_id_env_key = f"OPENAI_VECTOR_STORE_ID_{name.upper()}"
        with open('.env', 'a') as env_file:
            env_file.write(f'\n{vector_store_id_env_key}={vector_store.id}')
        
        

def vectorize_to_llama_index(data_path: str):
    pass






