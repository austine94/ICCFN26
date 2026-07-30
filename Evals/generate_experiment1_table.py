#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:37:36 2026

@author: localadmin
"""

import json
import pandas as pd

def generate_exp1_table(json_file='experiment_1_MR_results.json'):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}. Make sure you ran Experiment 1.")
        return

    df = pd.DataFrame(data)

    #calculate the mean accuracy per Model and Temperature
    summary_df = df.groupby(['temperature', 'model'])['accuracy'].mean().reset_index()
    summary_df['accuracy'] = summary_df['accuracy'] * 100
    
    pivot_table = summary_df.pivot(index='temperature', columns='model', values='accuracy')

    # Reorder columns to put the TF-IDF baseline first if it exists
    cols = list(pivot_table.columns)
    baseline_col = 'tfidf-cosine-baseline'
    if baseline_col in cols:
        cols.remove(baseline_col)
        # Sort the SLMs alphabetically, but keep the baseline at the far left
        cols = [baseline_col] + sorted(cols)
        pivot_table = pivot_table[cols]

    print("\n### Experiment 1: Accuracy vs. Temperature Matrix ###")
    print("Values represent: Average Accuracy (%)\n")
    
    # Fill NaN values (since TF-IDF only ran at Temp 0.0) with a dash
    markdown_table = pivot_table.fillna('-').to_markdown(floatfmt=".1f")
    print(markdown_table)

    output_csv = 'experiment_1_temperature_matrix.csv'
    # Fill NA before saving to CSV so the raw file is also clean
    pivot_table.fillna('-').to_csv(output_csv)
    print(f"\nSuccess! Table saved to {output_csv}")

if __name__ == "__main__":
    generate_exp1_table()