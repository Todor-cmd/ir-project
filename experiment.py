#TODO: Implement the file that will run all rags and evaluate results, saving the results to the results directory.
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_core.language_models.llms import BaseLLM
from ragas import EvaluationDataset

def generate_evaluations(rag, sample_queries, expected_responses, save_path):
    llm = BaseLLM() # TODO: Implement the LLM to use as judge for evaluation

    evaluator_llm = LangchainLLMWrapper(llm)

    dataset = []

    for query,reference in zip(sample_queries,expected_responses):

        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        dataset.append(
            {
                "user_input":query,
                "retrieved_contexts":relevant_docs,
                "response":response,
                "reference":reference
            }
        )
    
    # Save the raw dataset as CSV for reproducibility
    EvaluationDataset.from_list(dataset).to_csv(f"{save_path}/raw_dataset.csv")

    evaluation_dataset = EvaluationDataset.from_list(dataset) 

    evaluations = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
    evaluations.to_csv(f"{save_path}/evaluation_results.csv")



