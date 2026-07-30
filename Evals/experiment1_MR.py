#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:17:17 2026

@author: localadmin
"""

import json
import random
import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set fixed seed for dataset reproducibility
SEED = 100
random.seed(SEED)

def run_tfidf_baseline(test_samples):
    '''
    Evaluates the non-AI classical NLP baseline using TF-IDF + Cosine Similarity.
    Ranks candidate descriptions by similarity to the target description and selects top 3.
    '''
    print("\n=========================================")
    print("Running Non-AI Baseline: TF-IDF + Cosine Similarity")
    print("=========================================")
    
    baseline_results = []
    
    for idx, sample in enumerate(test_samples):
        target_cve = sample["target_cve"]
        candidates = sample["candidates"]
        true_indices = sample["true_indices"]
        target_cwe = sample["target_cwe"]
        
        # Raw text descriptions
        target_text = target_cve[1]
        candidate_texts = [cand[1] for cand in candidates]
        
        # Combine target and candidates into corpus
        corpus = [target_text] + candidate_texts
        
        try:
            # TF-IDF Vectorization with English stop words removed
            vectorizer = TfidfVectorizer(stop_words='english', analyzer='char_wb')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Cosine similarity between target (index 0) and candidate vectors (indices 1 to 4)
            target_vec = tfidf_matrix[0]
            candidate_vecs = tfidf_matrix[1:]
            sim_scores = cosine_similarity(target_vec, candidate_vecs)[0]
            
            # Rank candidates 1-4 by descending similarity score (select top 3)
            # np.argsort returns ascending order indices, so reverse with [::-1]
            top_3_zero_based = np.argsort(sim_scores)[::-1][:3]
            predicted_matches = [int(i + 1) for i in top_3_zero_based]
            
        except Exception as e:
            print(f"  TF-IDF Baseline Failed on Iteration {idx+1}: {e}")
            predicted_matches = []
            
        # Calculate accuracy
        correct = len(set(predicted_matches).intersection(set(true_indices)))
        accuracy = correct / 3.0
        
        print(f"  Baseline Iter {idx+1} | True: {sorted(true_indices)} | Pred: {sorted(predicted_matches)} | Acc: {accuracy:.2f}")
        
        baseline_results.append({
            "model": "tfidf-cosine-baseline",
            "temperature": 0.0,
            "iteration": idx + 1,
            "target_cwe": target_cwe,
            "true_indices": true_indices,
            "predicted_indices": predicted_matches,
            "accuracy": accuracy
        })
        
    return baseline_results


def run_experiment_one_mr(iterations_per_config=5):
    '''
    Tests how well models can identify matching CVEs using a 
    Map-Reduce architecture (distilling descriptions before comparison),
    benchmarked against a non-AI TF-IDF baseline.
    '''
    print("Loading Dataset...")
    try:
        with open('cwe_cve_10k_mapping.json', 'r') as f:
            cve_data = json.load(f)
    except FileNotFoundError:
        print("Error: cwe_cve_10k_mapping.json not found.")
        return

    models = [
        "llama3.2:3b",
        "gemma3:4b",
        "phi4-mini",
        "llama3:8b",
        "smollm2:360m" 
    ]
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    valid_cwes = {cwe: cves for cwe, cves in cve_data.items() if len(cves) >= 4}
    cwe_list = list(valid_cwes.keys())
    
    if not cwe_list:
        print("Error: Not enough data. No CWE has >= 4 CVEs.")
        return

    # =========================================================================
    # PRE-GENERATE DATASET SAMPLES
    # Pre-sampling guarantees all SLMs and the TF-IDF baseline evaluate on 
    # the EXACT same test instances for fair comparison.
    # =========================================================================
    print(f"Pre-sampling {iterations_per_config} identical test instances...")
    test_samples = []
    for _ in range(iterations_per_config):
        target_cwe = random.choice(cwe_list)
        sampled_cves = random.sample(list(valid_cwes[target_cwe].items()), 4)

        target_cve = sampled_cves[0]
        match_cves = sampled_cves[1:4] 

        decoy_cwes = random.sample([c for c in cwe_list if c != target_cwe], 1)
        decoy_cves = []
        for dcwe in decoy_cwes:
            dcve = random.choice(list(valid_cwes[dcwe].items()))
            decoy_cves.append(dcve)

        candidates = match_cves + decoy_cves
        random.shuffle(candidates)

        true_indices = []
        for idx, candidate in enumerate(candidates):
            if candidate in match_cves:
                true_indices.append(idx + 1)
                
        test_samples.append({
            "target_cwe": target_cwe,
            "target_cve": target_cve,
            "candidates": candidates,
            "true_indices": true_indices
        })

    all_results = []

    # 1. RUN NON-AI TF-IDF BASELINE FIRST
    baseline_results = run_tfidf_baseline(test_samples)
    all_results.extend(baseline_results)

    # 2. RUN SLM EVALUATIONS ACROSS TEMPERATURE MATRIX
    print("\nStarting Experiment 1 MR: Map-Reduce + Temperature Matrix...")

    for model in models:
        for temp in temperatures:
            print(f"\n=========================================")
            print(f"Testing Model: {model} | Temp: {temp}")
            print(f"=========================================")

            for i, sample in enumerate(test_samples):
                target_cwe = sample["target_cwe"]
                target_cve = sample["target_cve"]
                candidates = sample["candidates"]
                true_indices = sample["true_indices"]

                # ==========================================
                # --- MAP PHASE (Distill all descriptions) ---
                # ==========================================
                map_prompt_template = "Extract the core vulnerability mechanism from this description in 1-2 short sentences.\n\nDescription: {}"

                try:
                    # 1. Distill Target
                    target_map_res = ollama.chat(
                        model=model, 
                        messages=[{'role': 'user', 'content': map_prompt_template.format(target_cve[1])}], 
                        options={'temperature': 0.0} # Keep distillation strictly factual
                    )
                    distilled_target = target_map_res['message']['content'].strip()

                    # 2. Distill Candidates
                    distilled_candidates = []
                    for cve_id, desc in candidates:
                        cand_map_res = ollama.chat(
                            model=model, 
                            messages=[{'role': 'user', 'content': map_prompt_template.format(desc)}], 
                            options={'temperature': 0.0}
                        )
                        distilled_candidates.append(cand_map_res['message']['content'].strip())
                except Exception as e:
                    print(f"  FAILED during Map Phase on {model}: {e}")
                    continue

                # ==========================================
                # --- REDUCE PHASE (Evaluate the matches) ---
                # ==========================================
                candidates_text = "\n".join([f"Candidate {idx+1}: {desc}" for idx, desc in enumerate(distilled_candidates)])
                
                system_prompt = """You are a cyber threat intelligence analyst.
Your task is to identify which 3 candidate vulnerabilities share the EXACT SAME underlying technical weakness (CWE) as the target vulnerability. Do not group them by software or product; group them ONLY by the specific vulnerability mechanism.

Respond ONLY in strict JSON format matching this schema:
{
    "target_weakness": "Brief 1-sentence identification of the target's core mechanism.",
    "matches": [integer, integer, integer]
}

EXAMPLE Output:
{
    "target_weakness": "SQL Injection vulnerability due to improper input sanitization.",
    "matches": [2, 4, 5]
}
"""
                
                user_prompt = f"""Target Vulnerability Mechanism:
{distilled_target}

Candidate Mechanisms:
{candidates_text}

Analyze the Target Vulnerability, then provide the numbers of the 3 Candidates that share the exact same underlying weakness in the JSON format."""
                
                try:
                    response = ollama.chat(
                        model=model,
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_prompt}
                        ],
                        format='json',
                        options={
                            'temperature': temp, # Apply the matrix temperature here
                            'num_predict': 150   # Increased to allow full JSON generation
                        }
                    )

                    # Evaluation step
                    try:
                        prediction = json.loads(response['message']['content'])
                        predicted_matches = prediction.get('matches', [])
                        predicted_matches = [int(x) for x in predicted_matches if str(x).isdigit()]
                    except:
                        predicted_matches = []

                    # Calculate score
                    correct_predictions = len(set(predicted_matches).intersection(set(true_indices)))
                    accuracy = correct_predictions / 3.0

                    print(f"  Iter {i+1} | True: {sorted(true_indices)} | Pred: {sorted(predicted_matches)} | Acc: {accuracy:.2f}")

                    all_results.append({
                        "model": model,
                        "temperature": temp,
                        "iteration": i+1,
                        "target_cwe": target_cwe,
                        "true_indices": true_indices,
                        "predicted_indices": predicted_matches,
                        "accuracy": accuracy
                    })

                except Exception as e:
                    print(f"  FAILED during Reduce Phase on {model} at Temp {temp}: {e}")

    with open('experiment_1_MR_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    print("\nExperiment 1 MR complete! Results saved to experiment_1_MR_results.json")

if __name__ == "__main__":
    run_experiment_one_mr(iterations_per_config=200)