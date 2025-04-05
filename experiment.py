from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRecall, Faithfulness, ContextPrecision, FactualCorrectness, ContextEntityRecall, ResponseRelevancy
from ragas import EvaluationDataset

from langchain_groq import ChatGroq

from experimental_components.prepare_testsets import get_data_preparation_functions
from experimental_components.get_agents import get_rag_agents_for_experiment_1
import os
from pprint import pprint
import json
import pandas as pd
from tqdm import tqdm
    
def run_experiment(agents, data_preparation_functions):
    for dataset_name, data_preparation_function in data_preparation_functions.items():
        documents, queries, reference_responses, reference_contexts = data_preparation_function()
        
        # For bigger datasets, we only evaluate on the first 50 examples
        if dataset_name == "hotpotqa" or dataset_name == "nq":
            print(f"Shortening to use only first 50 examples of {dataset_name}")
            queries = queries[:50]
            
        
        for agent_name, agent in agents.items():
            print(f"Running experiment for {agent_name} on {dataset_name}")
            agent.load_documents(documents)
            
            save_path = f"./results/{dataset_name}/{agent_name}"
            
            # Reset dataset_to_evaluate for each agent
            dataset_to_evaluate = []
            
            print(f"Collecting responses from {agent_name} on {dataset_name}")
            for i in tqdm(range(len(queries))):
                query = queries[i]
                reference_response = reference_responses[i]
                reference_context = reference_contexts[i]
                
                agents_retrieved_docs = agent.get_most_relevant_docs(query)
                
                try:
                    agent_response = agent.generate_answer(query, agents_retrieved_docs)
                except Exception as e:
                    print(f"Error generating response for {agent_name} on {dataset_name} at query {i}: {e}")
                    break
                
                # Convert reference_response to string if it's a list
                if isinstance(reference_response, list):
                    reference_response = reference_response[0] if reference_response else ""
                
                dataset_to_evaluate.append(
                    {
                        "user_input": query,
                        "retrieved_contexts": agents_retrieved_docs,
                        "response": agent_response,
                        "reference": reference_response,
                        "reference_contexts": reference_context
                    }
                )
                
 
            eval_dataset = EvaluationDataset.from_list(dataset_to_evaluate)
            
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving raw evaluation dataset to {save_path}/raw_evaluation_dataset.csv")
           
            raw_eval_df = eval_dataset.to_pandas()
            raw_eval_df.to_csv(f"{save_path}/raw_evaluation_dataset.csv", encoding="utf-8", index=False)
            
            evaluator_llm = LangchainLLMWrapper(
                ChatGroq(model="deepseek-r1-distill-qwen-32b", temperature=0.2)
            )
            
            evaluations = evaluate(
                dataset=eval_dataset,
                metrics=[ContextPrecision(),
                        ContextRecall(),
                        Faithfulness(),
                        FactualCorrectness(),
                        ContextEntityRecall(),
                        ResponseRelevancy()
                        ],
                llm=evaluator_llm
            )
            
            print(f"Saving evaluation results to {save_path}/evaluation_results.csv")
            eval_df = evaluations.to_pandas()
            eval_df.to_csv(f"{save_path}/evaluation_results.csv", encoding="utf-8", index=False)
            
            print(f"Evaluation results for {agent_name} on {dataset_name}:")
            pprint(evaluations)
            
def finish_incomplete_evaluation(agent, dataset):
    #Get csv and make an EvaluationDataset from it
    save_path = f"./results/{dataset}/{agent}"
    path_to_csv = f"{save_path}/raw_evaluation_dataset.csv"
    
    df = pd.read_csv(path_to_csv)
    print("Original data:")
    print(df.head())
    
    # Check the format of first row to debug
    
    print("Sample retrieved_contexts value:", df["retrieved_contexts"].iloc[0])
    print("Type:", type(df["retrieved_contexts"].iloc[0]))
    
    # Make reference_contexts a list if it's a string representation
    df["reference_contexts"] = df["reference_contexts"].apply(lambda x: json.loads(x))
    
    # Handle retrieved_contexts based on its format
    def convert_to_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            import ast
            try:
                # Use ast.literal_eval instead of json.loads for Python string representations
                return ast.literal_eval(x)
            except (ValueError, SyntaxError) as e:
                print(f"Failed to parse with error: {e}")
                # If the string looks like a list but can't be parsed, 
                # try simple splitting approach
                if x.startswith('[') and x.endswith(']'):
                    content = x[1:-1]
                    # If it's a list of strings with quotes
                    if content.count("'") >= 2:
                        items = []
                        current = ""
                        in_quotes = False
                        for char in content:
                            if char == "'" and not in_quotes:
                                in_quotes = True
                            elif char == "'" and in_quotes:
                                in_quotes = False
                                items.append(current)
                                current = ""
                            elif in_quotes:
                                current += char
                            elif char == ',' and not in_quotes:
                                pass  # Skip commas between items
                        
                        if items:
                            return items
                return [x]  # Return as a single-item list if all else fails
        return []
    
    df["retrieved_contexts"] = df["retrieved_contexts"].apply(convert_to_list)
   
    print("\nParsed data:")
    print(df.head())
    
    eval_dataset = EvaluationDataset.from_pandas(df)
    
    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(model="deepseek-r1-distill-qwen-32b", temperature=0.2, max_tokens=4096)
    )
    
    #Evaluate the dataset
    evaluations = evaluate(
        dataset=eval_dataset,
        metrics=[ContextPrecision(),
                ContextRecall(),
                Faithfulness(),
                FactualCorrectness(),
                ContextEntityRecall(),
                ResponseRelevancy()
                ],
        llm=evaluator_llm
    )
            
    print(f"Saving evaluation results to {save_path}/evaluation_results.csv")
    eval_df = evaluations.to_pandas()
    eval_df.to_csv(f"{save_path}/evaluation_results.csv", encoding="utf-8", index=False)
    
    pprint(evaluations)

    
def run_experiment_1():
    agents = get_rag_agents_for_experiment_1()
    data_preparation_functions = get_data_preparation_functions()
    run_experiment(agents, data_preparation_functions)

if __name__ == "__main__":
    run_experiment_1()
    # agent = "openai"
    # dataset = "nq"
    # fnish_incomplete_evaluation(agent, dataset)
    