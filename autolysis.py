#!/usr/bin/env python
# /// Script for autolysis project
# Requires Python >= 3.9
# Dependencies:
# - pandas
# - numpy
# - matplotlib
# - seaborn
# - requests
# - python-dotenv
# - scikit-learn
# - scipy
# - fastapi
# - uvicorn

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import zscore
import scipy.cluster.hierarchy as sch
import argparse
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load dataset with encoding fallback
def load_dataset(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin-1")

# Clean and preprocess the dataset
def clean_and_analyze(dataset):
    # Drop rows with missing values
    cleaned_data = dataset.dropna()

    # Select only numeric columns for correlation and analysis
    numeric_data = cleaned_data.select_dtypes(include=[np.number])
    
    # Calculate basic statistics for numeric data
    stats = numeric_data.describe().transpose()
    
    # Detect missing values (still works for the original dataset)
    missing_values = dataset.isnull().sum()
    
    # Outliers using z-scores on numeric data
    z_scores = np.abs(zscore(numeric_data))
    outliers = (z_scores > 3).any(axis=1)
    
    # Correlation matrix (only for numeric data)
    correlation_matrix = numeric_data.corr()
    
    return cleaned_data, stats, missing_values, outliers, correlation_matrix

# Perform KMeans clustering
def cluster_data(dataset, output_plot):
    numeric_data = dataset.select_dtypes(include=['float64', 'int64']).dropna()
    if numeric_data.shape[1] < 2:
        return

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(numeric_data)
    dataset.loc[:, "Cluster"] = clusters  # Avoid SettingWithCopyWarning

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(numeric_data)

    # Adjusting figure size and dpi for smaller images
    plt.figure(figsize=(4, 4))  # Smaller size (4x4 inches)
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap="viridis")
    plt.title("KMeans Clustering - PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.savefig(output_plot, dpi=100)  # Reduced DPI for smaller image size
    plt.close()

# Perform Hierarchical Clustering
def hierarchical_clustering(dataset, output_plot):
    numeric_data = dataset.select_dtypes(include=['float64', 'int64']).dropna()
    if numeric_data.shape[1] < 2:
        return

    # Perform hierarchical clustering
    Z = sch.linkage(numeric_data, 'ward')

    # Adjusting figure size and dpi for smaller images
    plt.figure(figsize=(6, 4))  # Smaller size (6x4 inches)
    sch.dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.savefig(output_plot, dpi=100)  # Reduced DPI for smaller image size
    plt.close()

# Query OpenAI for insights using API Proxy
def query_llm(prompt, max_tokens=500, api_key=""):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error in LLM query: {str(e)}"

# Generate the storytelling prompt for OpenAI
def generate_storytelling_prompt(file_name, dataset, stats, missing_values, correlation_matrix):
    prompt = f"Analyze the dataset '{file_name}' with columns: {list(dataset.columns)}. Provide a detailed story about the data, including insights into trends, patterns, and anomalies.\n"
    prompt += f"Here are the summary statistics:\n{stats}\n\n"
    prompt += f"Missing values in the dataset:\n{missing_values}\n\n"
    prompt += f"Correlation matrix:\n{correlation_matrix}\n\n"
    prompt += "Tell a detailed story, highlighting interesting trends, potential outliers, and explaining any significant findings.\n"
    prompt += "Also, provide suggestions for further analysis or possible future actions based on the data."

    return prompt

# Main function to process the dataset from the command line
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process and analyze a dataset.")
    parser.add_argument("file", type=str, help="The CSV file to process")
    parser.add_argument("--api_key", type=str, help="The API key for OpenAI")

    args = parser.parse_args()

    # Get the AIProxy API key from the .env file if not provided
    ai_proxy_token = os.getenv("AIPROXY_TOKEN", args.api_key)

    # Load and process the dataset
    dataset = load_dataset(args.file)
    cleaned_data, stats, missing_values, outliers, correlation_matrix = clean_and_analyze(dataset)

    # Create output directory for the results
    output_dir = os.path.join(os.getcwd(), os.path.splitext(args.file)[0])
    os.makedirs(output_dir, exist_ok=True)

    # Save the correlation heatmap
    plt.figure(figsize=(4, 4))  # Adjust size of the heatmap (4x4 inches)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=90)  # Save with lower DPI for smaller size
    plt.close()

    # Run clustering analysis
    cluster_data(cleaned_data, os.path.join(output_dir, "cluster_plot.png"))

    # Run hierarchical clustering
    hierarchical_clustering(cleaned_data, os.path.join(output_dir, "hierarchical_plot.png"))

    # Generate the storytelling prompt for OpenAI
    prompt = generate_storytelling_prompt(args.file, dataset, stats, missing_values, correlation_matrix)

    # Query OpenAI for insights
    insights = query_llm(prompt, api_key=ai_proxy_token)

    # Save the insights to a README file
    with open(os.path.join(output_dir, "README.md"), "w") as report:
        report.write(f"# Dataset Analysis Story\n\n")
        report.write(f"Dataset contains {len(dataset)} rows and {len(dataset.columns)} columns.\n")
        report.write(f"Missing values:\n{missing_values}\n")
        report.write(f"Outliers detected: {outliers.sum()}\n")
        report.write("## Insights\n")
        report.write(insights)
        report.write("\n\n## Visualizations\n")
        report.write("![Correlation Heatmap](correlation_heatmap.png)\n")
        report.write("![Cluster Plot](cluster_plot.png)\n")
        report.write("![Hierarchical Plot](hierarchical_plot.png)\n")

    print(f"Processing completed for {args.file}")

if __name__ == "__main__":
    main()
