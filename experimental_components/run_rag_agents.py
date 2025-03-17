from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langgraph.graph import Graph
import os
from langchain_core.messages import HumanMessage

def run_openai_agent( assistant_id: str, questions: list[str]):
    """
    Runs an OpenAI assistant to answer a list of questions using RAG capabilities. It might be
    possible to use this to get the retrieved documents as well.

    This function processes each question through an OpenAI assistant that has access to a vector store.
    For each question, it:
    1. Creates a new thread
    2. Adds the question as a user message
    3. Runs the assistant and waits for completion
    4. Retrieves and stores the assistant's response

    Args:
        questions (list[str]): List of questions to be answered by the assistant

    Returns:
        list[str]: List of responses from the assistant, one for each input question

    Raises:
        Exception: If any run fails to complete successfully

    """
    responses=[]
    
    client = OpenAI()
    for question in questions:
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=question
        )
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_id
        )
        if run.status == "completed":
            # TODO: Possibly can use this to get the retrieved documents
            run_steps = client.beta.threads.runs.steps.list(
                thread_id=thread.id,
                run_id=run.id
            )
            
            
            # Store response
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            responses.append(messages.data[0].content[0].text.value)
        else:
            raise Exception(f"Run {run.id} failed with status {run.status}")
        
    return responses

def run_custom_agent(agent: Graph, questions: list[str]):
    """
    Runs a custom agent on a list of questions and returns the responses.

    This function:
    1. Takes a custom agent graph and list of questions
    2. Invokes the agent on each question
    3. Collects and returns all responses

    Args:
        agent (Graph): The compiled custom agent graph to use for answering questions. It's final output
        is a response to the question but we can modify it to return the retrieved documents as well.
        questions (list[str]): List of questions to be answered by the agent

    Returns:
        list[str]: List of responses from the agent, one for each input question
    """
    responses=[]
    for question in questions:
        response = agent.invoke({"messages": [HumanMessage(content=question)]})
        responses.append(response)
    return responses