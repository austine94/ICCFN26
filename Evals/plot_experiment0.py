#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:47:41 2026

@author: localadmin
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def generate_experiment_0_charts():
    #read in eval and hullucination data
    try:
        with open('experiment_0_evaluations.json', 'r') as f:
            eval_data = json.load(f)
        with open('experiment_0_hallucinations.json', 'r') as f:
            hallucination_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Make sure you have run both Gemini judge scripts first!")
        return

    #process evaluations
    model_scores = {}
    
    for iteration, data in eval_data.items():
        evals = data.get("gemini_evaluations", [])
        for ev in evals:
            model = ev.get("model")
            score = ev.get("score")
            
            if model and score is not None:
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(score)

    avg_scores = {model: np.mean(scores) for model, scores in model_scores.items()}
    
    #sort models by highest score first
    sorted_score_models = sorted(avg_scores.keys(), key=lambda x: avg_scores[x], reverse=True)
    sorted_scores = [avg_scores[m] for m in sorted_score_models]

    #process hallcuinations
    model_hallucinations = {}
    
    for iteration, data in hallucination_data.items():
        checks = data.get("hallucination_checks", [])
        for check in checks:
            model = check.get("model")
            has_hal = check.get("has_hallucination")
            
            if model and has_hal is not None:
                if model not in model_hallucinations:
                    model_hallucinations[model] = []
                model_hallucinations[model].append(1 if has_hal else 0)

    #calculate rate (0 to 100)
    hal_rates = {model: (np.mean(hals) * 100) for model, hals in model_hallucinations.items()}
    
    #sort models by lowest hallucination rate first
    sorted_hal_models = sorted(hal_rates.keys(), key=lambda x: hal_rates[x])
    sorted_rates = [hal_rates[m] for m in sorted_hal_models]

    #plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_score_models, sorted_scores, color='#4C72B0', edgecolor='black')
    
    plt.ylabel('Average Score (out of 10, LLM-as-Judge)', fontsize=16)
    plt.xlabel('Model', fontsize=16)
    plt.ylim(0, 10.5)
    
    plt.xticks(rotation=15, ha='right', fontsize=16)
    
    #add data labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        # Increased text label font size
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.1f}', ha='center', va='bottom', fontsize=14)
        
    plt.tight_layout()
    plt.savefig('exp0_average_scores.png', dpi=300)
    print("Saved 'exp0_average_scores.png'")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_hal_models, sorted_rates, color='#C44E52', edgecolor='black')
    
    plt.ylabel('Hallucination Rate (%)', fontsize=16)
    plt.xlabel('Model', fontsize=16)
    plt.ylim(0, max(sorted_rates + [10]) * 1.2) # Ensure headroom for labels
    
    plt.xticks(rotation=15, ha='right', fontsize=16)
    
    #add data labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(sorted_rates)*0.02), f'{yval:.1f}%', ha='center', va='bottom', fontsize=14)
        
    plt.tight_layout()
    plt.savefig('exp0_hallucination_rates.png', dpi=300)
    print("Saved 'exp0_hallucination_rates.png'")
    plt.close()

if __name__ == "__main__":
    generate_experiment_0_charts()