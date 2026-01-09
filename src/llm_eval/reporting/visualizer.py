import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def generate_radar_chart(df: pd.DataFrame, output_path: Path):
    # Calculate means for the metrics
    metrics = [c for c in df.columns if c not in ['query', 'response', 'reasoning']]
    values = df[metrics].mean().tolist()
    
    # Complete the circle
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='skyblue', alpha=0.4)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    plt.title("Model Performance Overview")
    plt.savefig(output_path)
    plt.close()

def generate_score_histograms(df: pd.DataFrame, output_dir: Path):
    metrics = [c for c in df.columns if c not in ['query', 'response', 'reasoning']]
    for metric in metrics:
        plt.figure(figsize=(8, 4))
        df[metric].hist(bins=10, color='teal', edgecolor='black')
        plt.title(f"Score Distribution: {metric}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig(output_dir / f"dist_{metric.lower()}.png")
        plt.close()