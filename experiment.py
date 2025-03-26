from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, LLMContextPrecisionWithoutReference, FactualCorrectness, ContextEntityRecall, NoiseSensitivity, ResponseRelevancy
from langchain_core.language_models.llms import BaseLLM
from ragas import EvaluationDataset
from datasets import load_dataset
import os
from experimental_components.custom_agent import CustomAgent
from experimental_components.custom_retrievers import HybridBM25Retriever
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
        supporting_facts: List of supporting fact documents for each query
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset("hotpot_qa", "distractor", split="train", trust_remote_code=True)
    
    # Take the top n samples
    samples = dataset.select(range(num_samples))
    
    sample_docs = []
    sample_queries = []
    expected_responses = []
    supporting_facts_list = []
    
    # Extract data from each sample
    for sample in samples:
        # Extract documents from context
        docs = []
        doc_ids = {}  # Map title to document index
        
        # The context is a dictionary with 'title' and 'sentences' keys
        titles = sample["context"]["title"]
        sentences_lists = sample["context"]["sentences"]
        
        for i, (title, sentences) in enumerate(zip(titles, sentences_lists)):
            doc_content = f"Title: {title}\n"
            doc_content += "\n".join(sentences)
            docs.append(doc_content)
            doc_ids[title] = i
        
        # Add all documents to our collection
        sample_docs.extend(docs)
        
        # Add the question
        sample_queries.append(sample["question"])
        
        # Add the expected answer
        expected_responses.append(sample["answer"])
        
        # Extract supporting facts
        supporting_facts = []
        for title, sent_id in zip(sample["supporting_facts"]["title"], sample["supporting_facts"]["sent_id"]):
            if title in doc_ids:
                doc_index = doc_ids[title]
                # Get the specific document that contains this supporting fact
                doc = docs[doc_index]
                supporting_facts.append(doc)
        
        supporting_facts_list.append(supporting_facts)
    
    return sample_docs, sample_queries, expected_responses, supporting_facts_list

def generate_evaluations(rag, sample_docs, sample_queries, expected_responses, supporting_facts_list, save_path):
    """
    Generate evaluations for a RAG system using RAGAS metrics, testing both all context and supporting facts.
    
    Args:
        rag: The RAG system to evaluate
        sample_docs: List of document strings
        sample_queries: List of query strings
        expected_responses: List of expected response strings
        supporting_facts_list: List of supporting fact documents for each query
        save_path: Path to save evaluation results
    """
    # Initialize evaluation LLM
    llm = ChatOpenAI(model="gpt-4o")
    evaluator_llm = LangchainLLMWrapper(llm)

    # First approach: evaluate using all context documents
    all_context_dataset = []
    
    rag.load_documents(sample_docs)

    for query, reference in zip(sample_queries, expected_responses):
        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        
        all_context_dataset.append(
            {
                "user_input": query,
                "retrieved_contexts": relevant_docs,
                "response": response,
                "reference": reference
            }
        )
    
    # Save the raw dataset as CSV for reproducibility
    all_context_eval_dataset = EvaluationDataset.from_list(all_context_dataset)
    all_context_eval_dataset.to_csv(f"{save_path}/all_context_raw_dataset.csv")

    # Run evaluation with all context documents
    all_context_evaluations = evaluate(
        dataset=all_context_eval_dataset,
        metrics=[LLMContextPrecisionWithoutReference(),
                 LLMContextRecall(),
                 Faithfulness(),
                 FactualCorrectness(),
                 ContextEntityRecall(),
                 NoiseSensitivity(),
                 ResponseRelevancy()],
        llm=evaluator_llm
    )
    
    print("All context evaluation results:")
    print(all_context_evaluations)
    
    # Save the evaluation results
    all_context_eval_df = pd.DataFrame([all_context_evaluations])
    all_context_eval_df.to_csv(f"{save_path}/all_context_evaluation_results.csv", index=False)
    
    # Second approach: evaluate using only supporting facts
    supporting_facts_dataset = []
    
    for i, (query, reference, supporting_facts) in enumerate(zip(sample_queries, expected_responses, supporting_facts_list)):
        # Load just the supporting facts for this query
        rag.load_documents(supporting_facts)
        
        # Get relevant docs and generate answer
        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        
        supporting_facts_dataset.append(
            {
                "user_input": query,
                "retrieved_contexts": relevant_docs,
                "response": response,
                "reference": reference
            }
        )
    
    # Save the supporting facts dataset
    supporting_facts_eval_dataset = EvaluationDataset.from_list(supporting_facts_dataset)
    supporting_facts_eval_dataset.to_csv(f"{save_path}/supporting_facts_raw_dataset.csv")

    # Run evaluation with supporting facts
    supporting_facts_evaluations = evaluate(
        dataset=supporting_facts_eval_dataset,
        metrics=[LLMContextPrecisionWithoutReference(),
                 LLMContextRecall(),
                 Faithfulness(),
                 FactualCorrectness()],
        llm=evaluator_llm
    )
    
    print("Supporting facts evaluation results:")
    print(supporting_facts_evaluations)
    
    # Save the supporting facts evaluation results
    supporting_facts_eval_df = pd.DataFrame([supporting_facts_evaluations])
    supporting_facts_eval_df.to_csv(f"{save_path}/supporting_facts_evaluation_results.csv", index=False)
    
    # Compare the results
    print("Comparison of evaluation results:")
    comparison = pd.DataFrame({
        "Metric": all_context_eval_df.columns,
        "All Context": all_context_eval_df.iloc[0].values,
        "Supporting Facts": supporting_facts_eval_df.iloc[0].values
    })
    comparison.to_csv(f"{save_path}/evaluation_comparison.csv", index=False)
    print(comparison)
    
    print(f"Evaluation results saved to {save_path}")

def run_experiment():
    """Run the experiment with HotpotQA dataset"""
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare dataset samples
    sample_docs, sample_queries, expected_responses, supporting_facts_list = prepare_hotpotqa_samples(5)
    
    # Initialize the RAG system (using OpenAI in this example)
    rag = CustomAgent(
        llm=ChatOpenAI(model="gpt-4o"),
        retriever=HybridBM25Retriever()
    )
    
    # Run evaluations
    generate_evaluations(
        rag=rag, 
        sample_docs=sample_docs, 
        sample_queries=sample_queries, 
        expected_responses=expected_responses, 
        supporting_facts_list=supporting_facts_list,
        save_path=results_dir
    )
    
    print(f"Evaluation results saved to {results_dir}")

if __name__ == "__main__":
    run_experiment()