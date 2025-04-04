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
        

        
        dataset_to_evaluate = []
        for agent_name, agent in agents.items():
            print(f"Getting responses for {agent_name} on {dataset_name}")
            if dataset_name == "sse_single" or dataset_name == "sse_multi":
                print(f"Skipping {agent_name} for {dataset_name} because it's done.")
                continue
            
            save_path = f"./results/{dataset_name}/{agent_name}"
            if not (dataset_name == "hotpotqa" and agent_name == "pinecone"):
                agent.load_documents(documents)
                
            for i in tqdm(range(len(queries))):
                query = queries[i]
                reference_response = reference_responses[i]
                reference_context = reference_contexts[i]
                
                agents_retrieved_docs = agent.get_most_relevant_docs(query)
                agent_response = agent.generate_answer(query, agents_retrieved_docs)
                
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
            
def fnish_incomplete_evaluation():
    #Get csv and make an EvaluationDataset from it
    save_path = f"./results/sse_single/hybrid"
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
        ChatGroq(model="deepseek-r1-distill-qwen-32b", temperature=0.2)
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
    # fnish_incomplete_evaluation()