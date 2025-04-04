import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from typing import Optional, List, Dict, Any
import glob


def visualize_results(filepath: str, agent_name: str = None, dataset_name: str = None, output_dir: Optional[str] = None) -> None:
    """
    Visualize results from a raw evaluation dataset CSV file.
    
    Args:
        filepath: Path to the raw_evaluation_dataset.csv file
        agent_name: Name of the agent (if not provided, will try to extract from filepath)
        dataset_name: Name of the dataset (if not provided, will try to extract from filepath)
        output_dir: Directory to save visualizations (defaults to same directory as input file)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract agent and dataset names from filepath if not provided
    if agent_name is None or dataset_name is None:
        parts = filepath.split('/')
        if len(parts) >= 3 and 'results' in parts:
            results_index = parts.index('results')
            if len(parts) > results_index + 2:
                if dataset_name is None:
                    dataset_name = parts[results_index + 1]
                if agent_name is None:
                    agent_name = parts[results_index + 2]
    
    # Set defaults if still not available
    if dataset_name is None:
        dataset_name = "unknown_dataset"
    if agent_name is None:
        agent_name = "unknown_agent"
    
    # Load the data
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Basic statistics
    print(f"Dataset: {dataset_name}, Agent: {agent_name}")
    print(f"Number of queries: {len(df)}")
    
    # Visualize query lengths
    plt.figure(figsize=(10, 6))
    df['query_length'] = df['user_input'].apply(len)
    sns.histplot(df['query_length'], bins=20, kde=True)
    plt.title(f'Distribution of Query Lengths - {agent_name} on {dataset_name}')
    plt.xlabel('Query Length (characters)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, f'{agent_name}_{dataset_name}_query_lengths.png'))
    
    # Visualize response lengths
    plt.figure(figsize=(10, 6))
    df['response_length'] = df['response'].apply(len)
    sns.histplot(df['response_length'], bins=20, kde=True)
    plt.title(f'Distribution of Response Lengths - {agent_name} on {dataset_name}')
    plt.xlabel('Response Length (characters)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, f'{agent_name}_{dataset_name}_response_lengths.png'))
    
    # Visualize number of retrieved contexts
    plt.figure(figsize=(10, 6))
    
    # Handle different formats of retrieved_contexts
    def count_contexts(contexts):
        if isinstance(contexts, list):
            return len(contexts)
        elif isinstance(contexts, str):
            try:
                # Try to parse as a list
                import ast
                parsed = ast.literal_eval(contexts)
                if isinstance(parsed, list):
                    return len(parsed)
                return 1
            except:
                # If it can't be parsed, count it as one context
                return 1
        return 0
    
    df['num_retrieved_contexts'] = df['retrieved_contexts'].apply(count_contexts)
    sns.countplot(x='num_retrieved_contexts', data=df)
    plt.title(f'Number of Retrieved Contexts - {agent_name} on {dataset_name}')
    plt.xlabel('Number of Contexts')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, f'{agent_name}_{dataset_name}_num_contexts.png'))
    
    # If there's an evaluation_results.csv in the same directory, visualize those metrics too
    eval_filepath = os.path.join(os.path.dirname(filepath), 'evaluation_results.csv')
    if os.path.exists(eval_filepath):
        visualize_evaluation_results(eval_filepath, output_dir, dataset_name, agent_name)
    
    print(f"Visualizations saved to {output_dir}")


def visualize_evaluation_results(filepath: str, output_dir: str, dataset_name: str, agent_name: str) -> None:
    """
    Visualize metrics from evaluation_results.csv
    
    Args:
        filepath: Path to the evaluation_results.csv file
        output_dir: Directory to save visualizations
        dataset_name: Name of the dataset
        agent_name: Name of the agent
    """
    print(f"Loading evaluation results from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Get all metric columns (excluding any non-metric columns)
    non_metric_cols = ['user_input', 'retrieved_contexts', 'response', 'reference', 'reference_contexts']
    metric_cols = [col for col in df.columns if col not in non_metric_cols]
    
    if not metric_cols:
        print("No metric columns found in evaluation results")
        return
    
    # Create a summary of metrics
    metrics_summary = df[metric_cols].describe()
    print("\nMetrics Summary:")
    print(metrics_summary)
    
    # Visualize each metric
    plt.figure(figsize=(12, 8))
    df[metric_cols].mean().plot(kind='bar')
    plt.title(f'Average Metrics - {agent_name} on {dataset_name}')
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Assuming metrics are normalized between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{agent_name}_{dataset_name}_metrics_summary.png'))
    
    # Visualize distribution of each metric
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metric_cols):
        plt.subplot(2, (len(metric_cols) + 1) // 2, i + 1)
        sns.histplot(df[metric], bins=10, kde=True)
        plt.title(f'{metric}')
        plt.xlabel('Score')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{agent_name}_{dataset_name}_metrics_distribution.png'))


def create_agent_comparison_table(results_dir: str = "results", output_dir: Optional[str] = None) -> None:
    """
    Create a single table comparing average metrics across all agents and datasets.
    
    Args:
        results_dir: Directory containing the results (default: "results")
        output_dir: Directory to save the comparison table (defaults to results_dir)
    """
    if output_dir is None:
        output_dir = results_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all evaluation_results.csv files
    eval_files = glob.glob(os.path.join(results_dir, "**", "evaluation_results.csv"), recursive=True)
    
    if not eval_files:
        print("No evaluation results found")
        return
    
    # Initialize a list to store all results
    all_results = []
    
    # Process each evaluation file
    for eval_file in eval_files:
        # Convert path to use forward slashes
        eval_file = eval_file.replace('\\', '/')
        
        # Extract dataset and agent names from the path
        parts = eval_file.split('/')
        if len(parts) >= 3 and 'results' in parts:
            results_index = parts.index('results')
            if len(parts) > results_index + 2:
                dataset_name = parts[results_index + 1]
                agent_name = parts[results_index + 2]
                
                # Load the evaluation results
                df = pd.read_csv(eval_file)
                
                # Get metric columns
                non_metric_cols = ['user_input', 'retrieved_contexts', 'response', 'reference', 'reference_contexts']
                metric_cols = [col for col in df.columns if col not in non_metric_cols]
                
                if metric_cols:
                    # Calculate average metrics
                    avg_metrics = df[metric_cols].mean().to_dict()
                    
                    # Add dataset and agent information
                    for metric, value in avg_metrics.items():
                        result_row = {
                            'Dataset': dataset_name,
                            'Agent': agent_name,
                            'Metric': metric,
                            'Value': value
                        }
                        all_results.append(result_row)
    
    if not all_results:
        print("No valid results found")
        return
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(all_results)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "all_agents_comparison.csv").replace('\\', '/')
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved comparison table to {csv_path}")
    
    # Create a heatmap visualization
    plt.figure(figsize=(15, 10))
    
    # Create a single heatmap with all metrics
    pivot_df = comparison_df.pivot(index='Agent', columns='Metric', values='Value')
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='Reds', cbar=True)
    plt.title(f'Metric Comparison Across Agents for Dataset: {dataset_name}')
    plt.xlabel('Metrics')
    plt.ylabel('Agent')
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_path = os.path.join(output_dir, "all_agents_comparison_heatmap.png").replace('\\', '/')
    plt.savefig(heatmap_path)
    print(f"Saved comparison heatmap to {heatmap_path}")
    plt.close()


if __name__ == "__main__":
    agents = ["pinecone",""]
    agent_name = "pinecone"
    dataset_name = "sse_multi"
    dataset_file_path = f"results/{dataset_name}"
    file_path = f"{dataset_file_path}/{agent_name}"
    raw_file_path = f"{file_path}/raw_evaluation_dataset.csv"
    eval_file_path = f"{file_path}/evaluation_results.csv"
    output_dir = f"{file_path}/visualisations_per_agent_and_dataset"
    
    # Create comparison table for dataset
    create_agent_comparison_table(dataset_file_path, dataset_file_path)
    
    # Visualise the results
    # visualize_results(raw_file_path, agent_name, dataset_name, output_dir)
    # visualize_evaluation_results(eval_file_path, output_dir, dataset_name, agent_name)
