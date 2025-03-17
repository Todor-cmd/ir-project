#TODO: Implement the file that will run all rags and evaluate results, saving the results to the results directory.

from experimental_components.create_rag_agents import create_openai_assistant, create_custom_agent
from experimental_components.run_rag_agents import run_openai_agent, run_custom_agent
from typing import Tuple

def run_experiment(dataset_name: str, dataset_path: str, agents: list[Tuple[str, str]]):
    """Runs a RAG experiment comparing different agents on a dataset of questions.

    Args:
        dataset_name (str): Name of the dataset being used, used for saving results
        dataset_path (str): Path to the dataset containing questions and answers
        agents (list[Tuple[str, str]]): List of tuples containing (llm_model, retrieval_strategy) pairs
            to test. The llm_model specifies which LLM to use (e.g. "gpt-4"), while retrieval_strategy 
            specifies the retrieval method ("openai-assistant" or other custom retrievers)

    The function:
    1. Gets questions and answers from the dataset
    2. For each agent configuration:
        - Creates the appropriate agent
        - Runs the agent on all questions
        - Saves the outputs and answers
        - Evaluates and saves the results
    """
    # Get questions from dataset
    questions, answers = get_questions_and_answers_from_dataset(dataset_path)
    
    # Evaluate each agent in agents list
    for llm_model, retrieval_strategy in agents:
        # The openai assistant has a different interface to the other agents that use langgraph
        if retrieval_strategy == "openai-assistant":
            # Create the agent
            openai_assistant = create_openai_assistant(llm_model, dataset_name)
            
            # Run the experiment
            responses = run_openai_agent(openai_assistant.id, questions)
        else:
            # Create the agent
            custom_agent = create_custom_agent(llm_model, retrieval_strategy, dataset_name)
            
            # Run the experiment
            responses = run_custom_agent(custom_agent, questions)
            
            
        # Save outputs and answers
        save_outputs_and_answers(dataset_name, responses, answers)
        
        # Save the results
        evaluate_results(dataset_name, responses)
    
    

    

def get_questions_and_answers_from_dataset(dataset_path: str):
    #TODO: Implement the function that will get the questions from the dataset
    pass

def save_outputs_and_answers(dataset_name: str, openai_responses: list[str], answers: list[str]):
    #TODO: Implement the function that will save the outputs and answers to the results directory.
    pass

def evaluate_results(dataset_name: str, openai_responses: list[str]):
    #TODO: Implement the function that will evaluate the results and save them to the results directory.
    pass


