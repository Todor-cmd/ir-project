# Comparing Open-AI Assistants to custom RAG solutons
This repository is home to the source code of Group 30 for the course  DSAIT4050 Information Retrieval at TU Delft. All results and instructions on how to interact with the source code to reproduce our results are found here.

## Environment Setup
Insall dependencies via Conda:
```bash
conda env create -f environment.yml
```

Configure you '.env' with your api keys. The keys you need are:
- OPENAI_API_KEY
- OPENAI_VECTOR_STORE_ID
- PINECONE_API_KEY
- PINECONE_HOST
- PINECONE_INDEX_NAME
- GROQ_API_KEY

To get these keys if you don't already have them, you will need to refer to the following websites:
 - https://www.pinecone.io/
 - https://groq.com/
 - https://openai.com/

## Run experiments
To run the experiment simply run:
```bash
python experiment.py
```

## Results 
The results will be generated in the `results/` directory. 

*Disclaimer:* The initial experiments we ran, collected data accuratley but contained certain mistakes that cost extra irrelevant rows of data. We moved all this into `results/not_cleaned/` and using `notebooks/clean_results.ipynb` we extracted relevant rows and stored them in `results/cleaned/`. The explanations of errors and what caused them can also be found in the notebook.

## Generating Visualisation
To create the heatmaps and visualisations from our paper you can use `notebooks/create_visualization.ipynb`.

## Generating Synthetic Data
All code for generating synthetic data is contained in `notebooks/generate_synthetic_testset.ipynb`.