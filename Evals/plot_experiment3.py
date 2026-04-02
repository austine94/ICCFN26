#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 06:12:22 2026

@author: localadmin
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_evaluation_plots(json_file='experiment_3_results.json'):
    print(f"Loading data from {json_file}...")
    
    #read in data and place in suitable format for making plots
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
                'is_correct': int(entry['is_correct']), # Convert True/False to 1/0 for math
                'total_time_sec': entry['metrics']['total_time_sec'],
                'tokens_per_sec': entry['metrics']['overall_tokens_per_sec'],
                'map_time_sec': entry['metrics']['map_time_sec'],
                'reduce_time_sec': entry['metrics']['reduce_time_sec']
            })

    df = pd.DataFrame(rows)

    summary_df = df.groupby(['k', 'model']).agg(
        accuracy=pd.NamedAgg(column='is_correct', aggfunc=lambda x: x.mean() * 100),
        avg_total_time=pd.NamedAgg(column='total_time_sec', aggfunc='mean'),
        avg_tokens_per_sec=pd.NamedAgg(column='tokens_per_sec', aggfunc='mean')
    ).reset_index()

    
    sns.set_style("white")
    models = summary_df['model'].unique()
    palette = sns.color_palette("bright", len(models))
    
    #Accuracy vs. Number of CVEs (k)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='k', y='accuracy', hue='model', marker='o', linewidth=2.5, palette=palette)
    plt.title('SLM and Map Reduce Accuracy by Number of CVEs Evaluated', fontsize=14, pad=15)
    plt.xlabel('Number of CVEs (k)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(-5, 105) # Keep percentage scale fixed from 0 to 100
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plot_1_accuracy_vs_k.png', dpi=300)
    plt.close()

    #Response Time vs. Number of CVEs (k)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='k', y='avg_total_time', hue='model', marker='s', linewidth=2.5, palette=palette)
    plt.title('Average Response Time', fontsize=14, pad=15)
    plt.xlabel('Number of CVEs (k)', fontsize=12)
    plt.ylabel('Total Time (Seconds)', fontsize=12)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plot_2_total_time_vs_k.png', dpi=300)
    plt.close()

    #tokens Per Second vs. Number of CVEs (k)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='k', y='avg_tokens_per_sec', hue='model', marker='^', linewidth=2.5, palette=palette)
    plt.title('SLM Answer Speed (Tokens/Sec)', fontsize=14, pad=15)
    plt.xlabel('Number of CVEs (k)', fontsize=12)
    plt.ylabel('Tokens Per Second', fontsize=12)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plot_3_tokens_per_sec_vs_k.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_evaluation_plots()