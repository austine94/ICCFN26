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
    # Read in eval and hallucination data
    try:
        with open('experiment_0_evaluations.json', 'r') as f:
            eval_data = json.load(f)
        with open('experiment_0_hallucinations.json', 'r') as f:
            hallucination_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Make sure you have run both Gemini judge scripts first!")
        return

    # Process evaluations (Accuracy & Explainability)
    model_accuracy_scores = {}
    model_explainability_scores = {}
    
    for iteration, data in eval_data.items():
        evals = data.get("gemini_evaluations", [])
        for ev in evals:
            model = ev.get("model")
            score = ev.get("score")
            exp_score = ev.get("explainability_score")
            
            # Accuracy scores
            if model and score is not None:
                if model not in model_accuracy_scores:
                    model_accuracy_scores[model] = []
                model_accuracy_scores[model].append(score)
                
            # Explainability scores
            if model and exp_score is not None:
                if model not in model_explainability_scores:
                    model_explainability_scores[model] = []
                model_explainability_scores[model].append(exp_score)

    # 1. Average Accuracy Scores
    avg_accuracy = {model: np.mean(scores) for model, scores in model_accuracy_scores.items()}
    sorted_acc_models = sorted(avg_accuracy.keys(), key=lambda x: avg_accuracy[x], reverse=True)
    sorted_acc_scores = [avg_accuracy[m] for m in sorted_acc_models]

    # 2. Average Explainability Scores
    avg_explainability = {model: np.mean(scores) for model, scores in model_explainability_scores.items()}
    sorted_exp_models = sorted(avg_explainability.keys(), key=lambda x: avg_explainability[x], reverse=True)
    sorted_exp_scores = [avg_explainability[m] for m in sorted_exp_models]

    # Process Hallucinations
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

    # Calculate hallucination rate (0 to 100)
    hal_rates = {model: (np.mean(hals) * 100) for model, hals in model_hallucinations.items()}
    sorted_hal_models = sorted(hal_rates.keys(), key=lambda x: hal_rates[x])
    sorted_rates = [hal_rates[m] for m in sorted_hal_models]

    # ==========================================
    # Plot 1: Average Accuracy Score
    # ==========================================
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_acc_models, sorted_acc_scores, color='#4C72B0', edgecolor='black')
    
    plt.ylabel('Average Accuracy Score (out of 10)', fontsize=16)
    plt.xlabel('Model', fontsize=16)
    plt.ylim(0, 10.5)
    plt.xticks(rotation=15, ha='right', fontsize=16)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.1f}', ha='center', va='bottom', fontsize=14)
        
    plt.tight_layout()
    plt.savefig('exp0_average_scores.png', dpi=300)
    print("Saved 'exp0_average_scores.png'")
    plt.close()

    # ==========================================
    # Plot 2: Explainability / Readability Score
    # ==========================================
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_exp_models, sorted_exp_scores, color='#55A868', edgecolor='black')
    
    plt.ylabel('Explainability Score (out of 10)', fontsize=16)
    plt.xlabel('Model', fontsize=16)
    plt.ylim(0, 10.5)
    plt.xticks(rotation=15, ha='right', fontsize=16)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.1f}', ha='center', va='bottom', fontsize=14)
        
    plt.tight_layout()
    plt.savefig('exp0_explainability_scores.png', dpi=300)
    print("Saved 'exp0_explainability_scores.png'")
    plt.close()
    
    # ==========================================
    # Plot 3: Hallucination Rate
    # ==========================================
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_hal_models, sorted_rates, color='#C44E52', edgecolor='black')
    
    plt.ylabel('Hallucination Rate (%)', fontsize=16)
    plt.xlabel('Model', fontsize=16)
    plt.ylim(0, max(sorted_rates + [10]) * 1.2)
    plt.xticks(rotation=15, ha='right', fontsize=16)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(sorted_rates)*0.02), f'{yval:.1f}%', ha='center', va='bottom', fontsize=14)
        
    plt.tight_layout()
    plt.savefig('exp0_hallucination_rates.png', dpi=300)
    print("Saved 'exp0_hallucination_rates.png'")
    plt.close()

if __name__ == "__main__":
    generate_experiment_0_charts()