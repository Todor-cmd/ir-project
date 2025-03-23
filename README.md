# Comparing Open-AI Assistants to custom RAG solutons
This repository is home to the source code of the report by Group 30 for the course  DSAIT4050 Information Retrieval at TU Delft.

## Environment Setup
Insall dependencies via Conda:
```bash
conda env create -f environment.yml
```

Configure you '.env' with your api keys. The keys you need are:
- OPENAI_API_KEY
- OPENAI_VECTOR_STORE_ID

## Upload data to vector 
This experiment utilitses too vector stores. To vectorize the data to these stores so it can be accessed during the final experiments run:
```bash
python vectorize_data.py
```

## Run experiments
To run the experiments simply run:
```bash
python experiment.py
```

## Results 
The results will be generated in the results directory. And can to generate visualisations for comparisons also in the results directory run:
```bash
python generate_result_visualisations.py
```