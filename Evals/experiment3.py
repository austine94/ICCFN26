#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 06:33:25 2026

@author: localadmin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 2026

@author: localadmin
"""

import json
import random
import ollama
import re
from llama_index.core import StorageContext, Settings
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.embeddings.ollama import OllamaEmbedding
from cwe2.database import Database

def run_experiment_three_kg_only(iterations_per_k=5, k_max=10):
    '''
    Experiment 3 (Ablation Study): 
    Takes a set of k CVEs with a shared common CWE, retrieves their  
    descriptions from the KG, and passes them to the SLM without 
    any Map-Reduce to identify the shared CWE.
    '''
    
    Settings.llm = LlamaIndexOllama(model="llama3.2:3b", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    #load the KG etc
    storage_context = StorageContext.from_defaults(persist_dir="./cve_knowledge_graph")
    graph_store = storage_context.graph_store 
    with open('cwe_cve_10k_mapping.json', 'r') as f:
        cve_data = json.load(f)
    cwe_db = Database()

    models = [
        "llama3.2:3b",
        "gemma3:4b",
        "phi4-mini",
        "llama3:8b",
        "smollm2:360m"
    ]

    all_results = {}
    print("Starting Experiment 3: KG Only (Raw Descriptions)...")

    for k in range(3, k_max+1):
        all_results[str(k)] = []
        
        #screen out CWEs that do not have k valid entries
        valid_cwes = {cwe: cves for cwe, cves in cve_data.items() if len(cves) >= k}
        cwe_list = list(valid_cwes.keys())
        
        if not cwe_list:  
            print(f"Not enough data for k={k}. Skipping.")
            continue

        for i in range(iterations_per_k):
            print(f"\n--- Iteration {i+1}/{iterations_per_k} for k={k} ---")
            
            #identify target CWE
            target_cwe = random.choice(cwe_list)
            
            #sample k CVE IDs
            sampled_cve_ids = random.sample(list(valid_cwes[target_cwe].keys()), k)
            
            #retrieve exact descriptions direct from the Knowledge Graph store
            raw_descriptions = []
            for cve_id in sampled_cve_ids:
                node_rels = graph_store.get(cve_id) 
                description = "No description found in KG."
                if node_rels:
                    for triplet in node_rels:
                        if "HAS_DESCRIPTION" in triplet:
                            description = triplet[-1] 
                            break
                            
                raw_descriptions.append((cve_id, description))

            #choose against a set of possible CWEs
            available_decoys = [c for c in cwe_list if c != target_cwe]
            decoy_cwes = random.sample(available_decoys, min(4, len(available_decoys))) # Standardized to 9 decoys
            options_list = [target_cwe] + decoy_cwes
            random.shuffle(options_list)
            
            #fetch official definitions for the prompt using cwe2
            options_text = ""
            for cwe in options_list:
                try:
                    cwe_int = int(cwe.replace("CWE-", ""))
                    weakness = cwe_db.get(cwe_int)
                    options_text += f"- {cwe}: {weakness.name}\n"
                except:
                    options_text += f"- {cwe}: Definition unavailable\n"

            # Instead of MR, just use raw descriptions
            raw_text_block = "\n\n".join([f"{cve_id} Description:\n{desc}" for cve_id, desc in raw_descriptions])

            #evaluate per model
            for model in models:
                print(f"  Testing {model}...", end=" ", flush=True)
                
                try:
                    system_prompt = f"""You are a cyber threat intelligence analyst. 
                Identify the common underlying weakness (CWE) shared by these {k} vulnerabilities.
                
                Options:
                {options_text}
                
                Respond ONLY in strict JSON format matching this schema:
                {{"predicted_cwe": "CWE-ID"}}"""

                    user_prompt = f"What is the common CWE shared by these raw vulnerability descriptions?\n\n{raw_text_block}"

                    response = ollama.chat(
                        model=model,
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_prompt}
                        ],
                        format='json',
                        options={'temperature': 0.0,
                                 'seed' : 100}
                    )
                    
                    try:
                        prediction = json.loads(response['message']['content'])
                        predicted_cwe = prediction.get('predicted_cwe', 'JSON_PARSE_ERROR')
                    except:
                        predicted_cwe = 'FORMAT_ERROR'
                    
                    is_correct = bool(re.search(rf"\b{target_cwe}\b", predicted_cwe))
                    print(f"Result: {predicted_cwe} (Correct: {is_correct})")

                    all_results[str(k)].append({
                        "iteration": i+1,
                        "target_cwe": target_cwe,
                        "model": model,
                        "predicted_cwe": predicted_cwe,
                        "is_correct": is_correct
                    })

                except Exception as e:
                    print(f"FAILED: {e}")

    with open('experiment_3_KG_Only_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\nExperiment 3 complete! Results saved to experiment_3_KG_Only_results.json")

if __name__ == "__main__":
    run_experiment_three_kg_only(iterations_per_k=50, k_max=3)