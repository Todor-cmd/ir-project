import os
import dotenv
dotenv.load_dotenv()
from typing import List, Tuple, Any

# For PDF extraction
from langchain_community.document_loaders import PyPDFLoader

# Ragas components
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import (
    apply_transforms,
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
)
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset import TestsetGenerator

# LLM and embeddings
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings


def load_pdf_documents(pdf_dir: str) -> List[Any]:
    """
    Load all PDF files from a directory and convert them to documents.
    """
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            print(f"Loading {filename}...")
            try:
                loader = PyPDFLoader(file_path)
                # Split by page to manage large files better
                docs = loader.load_and_split()
                documents.extend(docs)
                print(f"  Added {len(docs)} pages from {filename}")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
    
    return documents


def create_sustainable_se_personas() -> List[Persona]:
    """
    Create personas relevant to sustainable software engineering.
    """
    persona_prepared_student = Persona(
        name="Prepared Computer Science Student",
        role_description="A university student learning about sustainable software engineering. Has a good general overview of topics, but intrested in the finer technical details."
    )

    persona_lazy_student = Persona(
        name="Lazy Computer Science Student",
        role_description="A computer science student that barely knows the content, hasn't attended lectures, but is making a start with his studies. In addition to less tecnical questions, is also interested in organisational matters."
    )

    persona_curious_developer = Persona(
        name="Professional Developer",
        role_description="A software developer in the industry looking to learn about green software practices to apply in their projects and convince management of their importance."
    )

    return [persona_prepared_student, persona_lazy_student, persona_curious_developer]


def generate_sustainable_se_testset(pdf_dir: str, output_file: str = "data/sse_testset.csv"):
    """
    Generate a testset for sustainable software engineering using PDF lecture slides.
    """
    # Step 1: Load PDF documents
    print("Loading PDF documents...")
    documents = load_pdf_documents(pdf_dir)
    print(f"Loaded {len(documents)} document chunks.")
    
    # Step 2: Set up Groq LLM and embedding models
    print("Setting up LLM and embedding models...")
    generator_llm = LangchainLLMWrapper(
        ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.2)
    )
    # Still using OpenAI for embeddings as Groq doesn't have embedding models
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    
    # Step 3: Create knowledge graph
    print("Creating knowledge graph...")
    kg = KnowledgeGraph()
    
    for doc in documents:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
            )
        )
    
    print(f"Created initial knowledge graph with {len(kg.nodes)} nodes.")
    
    # Step 4: Apply transforms
    print("Applying transforms to enhance knowledge graph...")
    transforms = [
        HeadlinesExtractor(llm=generator_llm, max_num=20),
        HeadlineSplitter(max_tokens=1500),
        KeyphrasesExtractor(llm=generator_llm)
    ]
    
    apply_transforms(kg, transforms=transforms)
    print(f"Knowledge graph now has {len(kg.nodes)} nodes after transforms.")
    
    # Step 5: Configure personas
    personas = create_sustainable_se_personas()
    
    # Step 6: Set up query synthesizers
    query_distribution = [
        (
            SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"),
            0.5,
        ),
        (
            SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"),
            0.5,
        ),
    ]
    
    # Step 7: Generate testset
    print("Generating testset...")
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
        persona_list=personas,
    )
    
    testset_size = 30  # Adjust based on your needs
    testset = generator.generate(testset_size=testset_size, query_distribution=query_distribution)
    
    # Save to CSV
    print(f"Saving {len(testset)} generated test samples to {output_file}...")
    testset.to_pandas().to_csv(output_file, index=False)
    print("Done!")
    
    return testset


if __name__ == "__main__":
    """
    Generates a synthetic testset for RAG evaluation using PDF lecture slides
    based on the Ragas testset generation approach.
    """

    # Replace with the actual path to your lecture slides
    pdf_directory = "data/sse_lectures"
    testset = generate_sustainable_se_testset(pdf_directory)
    
    # Display a few examples
    print("\nSample generated test questions:")
    for i, row in enumerate(testset.to_pandas().itertuples()):
        if i >= 5:  # Show only 5 examples
            break
        print(f"\nQuestion {i+1}: {row.user_input}")
        print(f"Answer: {row.reference[:200]}..." if len(row.reference) > 200 else row.reference)
