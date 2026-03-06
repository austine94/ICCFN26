#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 06:38:53 2026

@author: localadmin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for Experiment 3 Plotting
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_exp3_plots(json_file='experiment_3_KG_Only_results.json'):
    print(f"Loading data from {json_file}...")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}. Make sure the script is in the same directory.")
        return

    rows = []
    for k_str, iterations in data.items():
        k_val = int(k_str)
        for entry in iterations:
            rows.append({
                'k': k_val,
                'model': entry['model'],
                'is_correct': int(entry['is_correct']) # Convert True/False to 1/0 for math
            })

    df = pd.DataFrame(rows)

    summary_df = df.groupby(['k', 'model']).agg(
        accuracy=pd.NamedAgg(column='is_correct', aggfunc=lambda x: x.mean() * 100)
    ).reset_index()
    sns.set_style("white")
    models = summary_df['model'].unique()
    palette = sns.color_palette("bright", len(models))
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='k', y='accuracy', hue='model', marker='o', linewidth=2.5, palette=palette)
    plt.title('SLM Accuracy by Number of CVEs (KG Only Baseline)', fontsize=14, pad=15)
    plt.xlabel('Number of CVEs (k)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(-5, 105) 
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    sns.despine()
    
    plt.tight_layout()
    plt.savefig('plot_exp3_accuracy_vs_k.png', dpi=300)
    plt.close()
    
    print("Success! Saved the Experiment 3 Accuracy plot to your directory.")

if __name__ == "__main__":
    generate_exp3_plots()