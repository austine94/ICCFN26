#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import time
import ollama
from llama_index.core import StorageContext, Settings
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.embeddings.ollama import OllamaEmbedding

random.seed(100)

def run_experiment_zero(num_iterations=50):
    '''
    Establishes a baseline for reading comprehension and summarization.
    Passes a single CVE description to 5 SLMs and asks for a natural 
    language summary of the vulnerability mechanism and impact.
    
    This is named experiment 0 because it is illustrative, rather than a test
    of the model/system capabilities.
    '''
    
    # Ensure LlamaIndex doesn't look for OpenAI keys
    Settings.llm = LlamaIndexOllama(model="llama3.2:3b", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    storage_context = StorageContext.from_defaults(persist_dir="./cve_knowledge_graph")
    graph_store = storage_context.graph_store 
    
    #load the valid CVE list from our ground truth file
    with open('cwe_cve_10k_mapping.json', 'r') as f:
        cve_data = json.load(f)
        
    #flatten the dictionary to get a master list of all CVE IDs
    all_cve_ids = []
    for cwes, cves in cve_data.items():
        all_cve_ids.extend(list(cves.keys()))
        
    random.shuffle(all_cve_ids)

    models = [
        "llama3.2:3b",
        "gemma3:4b",
        "phi4-mini",
        "llama3:8b",
        "smollm2:360m"
    ]

    all_results = {}
    print(f"Starting Experiment 0: Baseline Summarization ({num_iterations} iterations)...")

    cve_index = 0
    iteration = 1

    while iteration <= num_iterations and cve_index < len(all_cve_ids):
        target_cve = all_cve_ids[cve_index]
        cve_index += 1
        
        # Query the KG for Description only
        node_rels = graph_store.get(target_cve)
        description = None
        
        if node_rels:
            for triplet in node_rels:
                if "HAS_DESCRIPTION" in triplet:
                    description = triplet[-1]
                    break # Found what we need
                    
        # Skip this CVE if the KG is missing the description
        if not description:
            continue
            
        print(f"\n--- Iteration {iteration}/{num_iterations} | Target: {target_cve} ---")
        
        iteration_data = {
            "cve_id": target_cve,
            "ground_truth_description": description,
            "model_responses": {}
        }
        
        prompt = f"""You are a technical cybersecurity analyst. 
        Read the following vulnerability description.
        Provide a 1-2 sentence summary explaining exactly what the vulnerability is and its potential impact.
        
        Description: {description}
        
        Summary:"""

        #evaluate each model
        for model in models:
            print(f"  Testing {model}...", end=" ", flush=True)
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'temperature': 0.0,
                        'seed': 100
                    }
                )
                
                model_summary = response['message']['content'].strip()
                iteration_data["model_responses"][model] = model_summary
                print("Done.")
                
            except Exception as e:
                iteration_data["model_responses"][model] = f"ERROR: {str(e)}"
                print("FAILED.")
                
        all_results[str(iteration)] = iteration_data
        iteration += 1

    with open('experiment_0_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\nExperiment 0 complete! Results saved to experiment_0_results.json")

if __name__ == "__main__":
    run_experiment_zero(num_iterations=100)