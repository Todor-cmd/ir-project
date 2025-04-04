from llama_index.readers.file import PyMuPDFReader
from .custom_agent import CustomAgent
import os
from datasets import load_dataset
import re
from llama_index.core import Document
import pickle
import pandas as pd
from pprint import pprint

def prepare_sse_testset(is_multihop=False):
    # Load all PDF documents from the data directory
    pdf_dir = "./data/sse_lectures"
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    reader = PyMuPDFReader()
    documents = []
    for pdf_file in pdf_files:
        docs = reader.load(file_path=pdf_file)
        # join all the documents into one
        doc_text = "\n".join([doc.text for doc in docs])
        documents.append(Document(text=doc_text))
        print(f"Loaded document from {pdf_file}")
        
    testset_path = "./data/sse_testsets/sse_single_hop_testset.csv"
    if is_multihop:
        print("Using multihop testset")
        testset_path = "./data/sse_testsets/sse_multihop_testset.csv"
        
    # Load the testset
    testset = pd.read_csv(testset_path)
    
    # Group by user_input and reference to handle multiple contexts
    grouped = testset.groupby(['user_input', 'reference'])
    
    queries = []
    reference_responses = []
    golden_contexts = []
    
    # Process each group
    for (query, ref), group in grouped:
        queries.append(query)
        reference_responses.append(ref)
        # Collect all reference contexts for this query-reference pair
        contexts = group['reference_contexts'].tolist()
        golden_contexts.append(contexts)
    
    print(f"Loaded {len(queries)} unique query-reference pairs")
    print(f"Total documents loaded: {len(documents)}")
    
    return documents, queries, reference_responses, golden_contexts

def prepare_nq_testset(num_samples=500, use_cache=True, cache_path="./data/nq_processed_data.pkl"):
    # Check if cached data exists
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("lighteval/natural_questions_clean", split="validation", trust_remote_code=True)
    
    samples = dataset.select(range(num_samples))
    
    documents = []
    queries = []
    reference_responses = []
    golden_contexts = []
    
    document_titles = set()
    
    for sample in samples:
        question = sample["question"]
        answer = sample["short_answers"]
        reference_contexts = sample["long_answers"]
        doc_title = sample["title"]
        doc_text = sample["document"]
        
        if doc_title not in document_titles:
            document_titles.add(doc_title)
            doc_text = f"Title: {doc_title}\n{doc_text}"
            documents.append(Document(text=doc_text))
            
        queries.append(question)
        reference_responses.append(answer)
        golden_contexts.append(reference_contexts)
        
    # After processing, save to cache
    if use_cache:
        print(f"Saving processed data to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((documents, queries, reference_responses, golden_contexts), f)
    
    return documents, queries, reference_responses, golden_contexts
        
        
def prepare_hotpotqa_testset(num_samples=500, use_cache=True, cache_path="./data/hotpotqa_processed_data.pkl"):
    # Check if cached data exists
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Load the dataset from Hugging Face
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    
    samples = dataset.select(range(num_samples))
    
    # Take the top n samples
    documents = []
    queries = []
    reference_responses = []
    golden_contexts = []
    
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
        
        # Make Document objects and add all documents to our collection
        docs = [Document(text=doc) for doc in docs]
        documents.extend(docs)
        
        # Add the question
        queries.append(sample["question"])
        
        # Add the expected answer
        reference_responses.append(sample["answer"])
        
        # Extract supporting facts
        supporting_facts = []
        for title, sent_id in zip(sample["supporting_facts"]["title"], sample["supporting_facts"]["sent_id"]):
            if title in doc_ids:
                doc_index = doc_ids[title]
                # Get the specific document that contains this supporting fact
                doc = docs[doc_index]
                supporting_facts.append(doc.text)
        
        golden_contexts.append(supporting_facts)
        
    # Merge 10 documents into one document
    merged_documents = []
    for i in range(0, len(documents), 10):
        merged_doc = "\n".join([doc.text for doc in documents[i:i+10]])
        merged_documents.append(Document(text=merged_doc))
        
    documents = merged_documents

    # After processing, save to cache
    if use_cache:
        print(f"Saving processed data to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((documents, queries, reference_responses, golden_contexts), f)
    
    return documents, queries, reference_responses, golden_contexts

def get_data_preparation_functions():
    return {
        "sse_single": lambda **kwargs: prepare_sse_testset(is_multihop=False, **kwargs),
        "sse_multi": lambda **kwargs: prepare_sse_testset(is_multihop=True, **kwargs),
        "hotpotqa": prepare_hotpotqa_testset,
        "nq": prepare_nq_testset
    }

if __name__ == "__main__":
    # Will use cached data if available, otherwise process and cache
    documents, queries, reference_responses, golden_contexts = prepare_sse_testset(is_multihop=True)
    print(len(documents))
    print(documents[2].text)
    

