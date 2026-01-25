#!/usr/bin/env python3
"""
Visualize results from local results directory.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_local_results(results_dir="results"):
    """Load all results from local results directory."""
    results_dir = Path(results_dir)
    all_results = {}
    
    # Iterate through all model directories
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name.replace("__", "/")
        
        # Iterate through revision directories
        for revision_dir in model_dir.iterdir():
            if not revision_dir.is_dir():
                continue
            
            revision = revision_dir.name
            
            # Load all JSON files (excluding model_meta.json)
            for json_file in revision_dir.glob("*.json"):
                if json_file.name == "model_meta.json":
                    continue
                
                task_name = json_file.stem
                
                with open(json_file, 'r') as f:
                    task_result = json.load(f)
                
                # Extract main score from test split
                if "scores" in task_result and "test" in task_result["scores"]:
                    test_scores = task_result["scores"]["test"]
                    if test_scores and len(test_scores) > 0:
                        main_score = test_scores[0].get("main_score")
                        if main_score is not None:
                            key = f"{model_name}__{revision}"
                            if key not in all_results:
                                all_results[key] = {}
                            all_results[key][task_name] = main_score
    
    return all_results

def create_visualizations(results_dict, output_dir="benchmark_tables"):
    """Create visualizations from results dictionary."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results_dict).T
    df.index.name = "Model"
    df = df.reset_index()
    
    # Split model name and revision
    df[['Model', 'Revision']] = df['Model'].str.split('__', n=1, expand=True)
    
    # Group by model and calculate means
    model_stats = []
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        stats = {
            'Model': model,
            'Num Tasks': model_df.iloc[:, 2:].notna().sum(axis=1).max(),  # Count non-null tasks
        }
        
        # Calculate means for each task type
        task_columns = [col for col in df.columns if col not in ['Model', 'Revision']]
        for col in task_columns:
            values = model_df[col].dropna()
            if len(values) > 0:
                stats[col] = values.mean()
        
        model_stats.append(stats)
    
    stats_df = pd.DataFrame(model_stats)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "MTEB(Turkish)_summary.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")
    
    # Create visualizations
    task_columns = [col for col in stats_df.columns if col not in ['Model', 'Num Tasks']]
    
    if len(task_columns) > 0:
        # 1. Overall performance bar chart
        plt.figure(figsize=(12, 6))
        mean_scores = stats_df[task_columns].mean(axis=1)
        stats_df_sorted = stats_df.iloc[mean_scores.sort_values(ascending=False).index]
        
        plt.barh(stats_df_sorted['Model'], mean_scores.sort_values(ascending=False))
        plt.xlabel('Mean Score Across Tasks')
        plt.title('Overall Performance by Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "overall_performance.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved overall performance chart to {os.path.join(output_dir, 'overall_performance.png')}")
        
        # 2. Task-specific performance heatmap
        if len(task_columns) > 1:
            plt.figure(figsize=(max(12, len(task_columns) * 1.5), max(6, len(stats_df) * 0.5)))
            heatmap_data = stats_df.set_index('Model')[task_columns]
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm', cbar_kws={'label': 'Score'})
            plt.title('Task-Specific Performance Heatmap')
            plt.ylabel('Model')
            plt.xlabel('Task')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "task_performance_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved heatmap to {os.path.join(output_dir, 'task_performance_heatmap.png')}")
        
        # 3. Task performance comparison
        plt.figure(figsize=(14, 7))
        df_melted = stats_df.melt(id_vars=['Model'], value_vars=task_columns, 
                                  var_name='Task', value_name='Score')
        df_melted = df_melted.dropna()
        
        if len(df_melted) > 0:
            sns.boxplot(x='Task', y='Score', data=df_melted, palette='Set2')
            plt.xticks(rotation=45, ha='right')
            plt.title('Performance Distribution Across Tasks')
            plt.xlabel('Task')
            plt.ylabel('Score')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "task_performance_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved task comparison chart to {os.path.join(output_dir, 'task_performance_comparison.png')}")
    
    return stats_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize local MTEB results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to results directory (default: results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_tables",
        help="Path to save visualizations (default: benchmark_tables)"
    )
    
    args = parser.parse_args()
    
    print("Loading results from local directory...")
    results = load_local_results(args.results_dir)
    
    if not results:
        print("No results found in the specified directory.")
        exit(1)
    
    print(f"Found results for {len(results)} model(s)")
    
    print("Creating visualizations...")
    stats_df = create_visualizations(results, args.output_dir)
    
    print("\nSummary Statistics:")
    print(stats_df.to_string())
    
    print("\nVisualization complete!")

