from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_core.language_models.llms import BaseLLM
from ragas import EvaluationDataset
from datasets import load_dataset
import os
from experimental_components.open_ai_agent import OpenAIAssistant
from langchain_openai import ChatOpenAI
import pandas as pd


def prepare_hotpotqa_samples(num_samples=5):
    """
    Load the HotpotQA dataset and prepare sample documents, queries, and expected responses.
    
    Args:
        num_samples: Number of samples to use from the dataset
        
    Returns:
        sample_docs: List of document strings
        sample_queries: List of query strings
        expected_responses: List of expected response strings
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset("hotpot_qa", "distractor", split="train", trust_remote_code=True)
    
    # Take the top n samples
    samples = dataset.select(range(num_samples))
    
    sample_docs = []
    sample_queries = []
    expected_responses = []
    
    # Extract data from each sample
    for sample in samples:
        # Extract documents from context
        docs = []
        # The context is a dictionary with 'title' and 'sentences' keys
        titles = sample["context"]["title"]
        sentences_lists = sample["context"]["sentences"]
        
        for title, sentences in zip(titles, sentences_lists):
            doc_content = f"Title: {title}\n"
            doc_content += "\n".join(sentences)
            docs.append(doc_content)
        
        # Add all documents to our collection
        sample_docs.extend(docs)
        
        # Add the question
        sample_queries.append(sample["question"])
        
        # Add the expected answer
        expected_responses.append(sample["answer"])
    
    return sample_docs, sample_queries, expected_responses

def generate_evaluations(rag, sample_docs, sample_queries, expected_responses, save_path):
    """
    Generate evaluations for a RAG system using RAGAS metrics.
    
    Args:
        rag: The RAG system to evaluate
        sample_docs: List of document strings
        sample_queries: List of query strings
        expected_responses: List of expected response strings
        save_path: Path to save evaluation results
    """
    # Initialize evaluation LLM
    llm = ChatOpenAI(model="gpt-4o")
    evaluator_llm = LangchainLLMWrapper(llm)

    dataset = []
    
    rag.load_documents(sample_docs)

    for query, reference in zip(sample_queries, expected_responses):
        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        dataset.append(
            {
                "user_input": query,
                "retrieved_contexts": relevant_docs,
                "response": response,
                "reference": reference
            }
        )
    
    # Save the raw dataset as CSV for reproducibility
    EvaluationDataset.from_list(dataset).to_csv(f"{save_path}/raw_dataset.csv")

    evaluation_dataset = EvaluationDataset.from_list(dataset) 

    evaluations = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm
    )
    
    # Based on the RAGAS documentation, the result is a simple dictionary
    print("Evaluation results:")
    print(evaluations)
    
    # Save the evaluation results as a simple CSV
    # Convert dictionary to a DataFrame with a single row
    eval_df = pd.DataFrame([evaluations])
    eval_df.to_csv(f"{save_path}/evaluation_results.csv", index=False)
    
    print(f"Evaluation results saved to {save_path}/evaluation_results.csv")

def run_experiment():
    """Run the experiment with HotpotQA dataset"""
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare dataset samples
    sample_docs, sample_queries, expected_responses = prepare_hotpotqa_samples(5)
    
    # Initialize the RAG system (using OpenAI in this example)
    rag = OpenAIAssistant(model="gpt-4o")
    
    # Run evaluations
    generate_evaluations(
        rag=rag, 
        sample_docs=sample_docs, 
        sample_queries=sample_queries, 
        expected_responses=expected_responses, 
        save_path=results_dir
    )
    
    print(f"Evaluation results saved to {results_dir}")

if __name__ == "__main__":
    run_experiment()