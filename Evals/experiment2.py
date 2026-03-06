#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 05:25:50 2026

@author: localadmin
"""

import json
import random
import time
import ollama
import re
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.embeddings.ollama import OllamaEmbedding
from cwe2.database import Database

def run_experiment_two_kg_mr(iterations_per_k=5, k_max = 10):
    '''
    This experiment takes a set of k CVEs with a shared common CWE, passes
    them to the SLM, which performs a Map Reduce on their descriptions,
    and then asks if it can identify (using the KG) the shared CWE.
    
    We perform iterations_per_k of each batch size, and go from k = 3 CVEs
    to K_max
    '''
    
    #ensure LlamaIndex doesn't look for OpenAI keys
    Settings.llm = LlamaIndexOllama(model="llama3.2:3b", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    #load the Knowledge Graph & Storage Context
    print("Loading Knowledge Graph from disk...")
    storage_context = StorageContext.from_defaults(persist_dir="./cve_knowledge_graph")
    graph_store = storage_context.graph_store 
    
    #load the Ground Truth JSON and CWE Database
    with open('cwe_cve_10k_mapping.json', 'r') as f:
        cve_data = json.load(f)
    cwe_db = Database()

    #set of models to evaluate
    models = [
        "llama3.2:3b",
        "gemma3:4b",
        "phi4-mini",
        "llama3:8b",
        "smollm2:360m"
    ]

    all_results = {}
    print("Starting Experiment 2: KG + Map-Reduce...")

    for k in range(3, k_max+1):
        print(k)
        all_results[str(k)] = []
        #screen out CWEs that do not have k valid entries
        valid_cwes = {cwe: cves for cwe, cves in cve_data.items() if len(cves) >= k}
        cwe_list = list(valid_cwes.keys())
        
        if not cwe_list:  #if k is so large we have no CWE with enough CVEs
            print(f"Not enough data for k={k}. Skipping.")
            continue

        for i in range(iterations_per_k):
            
            #identify target CWE
            target_cwe = random.choice(cwe_list)
            print(target_cwe)
            #sample k CVE IDs
            sampled_cve_ids = random.sample(list(valid_cwes[target_cwe].keys()), k)
            
            #retrieve exact descriptions direct from the Knowledge Graph store
            raw_descriptions = []
            for cve_id in sampled_cve_ids:
                node_rels = graph_store.get(cve_id) 
                description = "No description found in KG."
                #this is a nightmare to get from llamaindex to the right format
                if node_rels:
                    #node_rels is a list of triplets: e.g., [['CVE-123', 'HAS_DESCRIPTION', 'Text...'], ...]
                    for triplet in node_rels:
                        #check if this specific triplet is the description edge
                        if "HAS_DESCRIPTION" in triplet:
                            description = triplet[-1] #object we need is the last item 
                            break
                            
                raw_descriptions.append((cve_id, description))

            #choose against a set of possible CWEs
            available_decoys = [c for c in cwe_list if c != target_cwe]
            decoy_cwes = random.sample(available_decoys, min(4, len(available_decoys)))
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

            #evaluate per model
            for model in models:
                print(f"  Testing {model}...", end=" ", flush=True)
                
                try:
                    #mapping into keywords
                    map_total_time = 0.0
                    map_total_tokens = 0
                    distilled_descriptions = []
                    
                    for cve_id, raw_desc in raw_descriptions:
                        map_prompt = f"""You are a technical cybersecurity analyst. Extract the core vulnerability mechanisms from the description in 1-3 sentences.
                        
                        Example Input: "A vulnerability in the login module of AppX allows remote attackers to inject arbitrary SQL commands via the 'user' parameter."
                        Example Output: SQL injection in login parameter.
                        
                        Input: "{raw_desc}"
                        Output:"""
                        
                        map_response = ollama.chat(
                            model=model,
                            messages=[{'role': 'user', 'content': map_prompt}],
                            options={'temperature': 0.0,
                                     'seed' : 100}
                        )
                                                
                        distilled_descriptions.append(f"{cve_id} Keywords: {map_response['message']['content']}")
                        #divide by 1e9 as Ollama is in nanoseconds
                        map_total_time += map_response.get('total_duration', 0) / 1e9
                        map_total_tokens += map_response.get('eval_count', 1)

                    #reduce step
                    distilled_text_block = "\n".join(distilled_descriptions)
                    
                    system_prompt = f"""You are a cyber threat intelligence analyst. 
                    Identify the common underlying weakness (CWE) shared by these {k} vulnerabilities.
                    
                    Options:
                    {options_text}
                    
                    Respond ONLY in strict JSON format matching this schema:
                    {{"predicted_cwe": "CWE-ID"}}"""

                    user_prompt = f"What is the common CWE shared by these condensed technical mechanisms?\n\n{distilled_text_block}"

                    reduce_response = ollama.chat(
                        model=model,
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_prompt}
                        ],
                        format='json',
                        options={'temperature': 0.0}
                    )
                    
                    reduce_time = reduce_response.get('total_duration', 0) / 1e9
                    reduce_tokens = reduce_response.get('eval_count', 1)
                    
                    #calculate metrics 
                    total_pipeline_time = map_total_time + reduce_time
                    total_tokens = map_total_tokens + reduce_tokens
                    overall_tps = total_tokens / total_pipeline_time if total_pipeline_time > 0 else 0
                    
                    #parse json to store
                    try:
                        prediction = json.loads(reduce_response['message']['content'])
                        predicted_cwe = prediction.get('predicted_cwe', 'JSON_PARSE_ERROR')
                    except:
                        predicted_cwe = 'FORMAT_ERROR'
                    
                    is_correct = bool(re.search(rf"\b{target_cwe}\b", predicted_cwe))
                    print(f"Result: {predicted_cwe} (Correct: {is_correct}) | Speed: {overall_tps:.1f} t/s | Time: {total_pipeline_time:.2f}s")

                    all_results[str(k)].append({
                        "iteration": i+1,
                        "target_cwe": target_cwe,
                        "model": model,
                        "predicted_cwe": predicted_cwe,
                        "is_correct": is_correct,
                        "metrics": {
                            "map_time_sec": map_total_time,
                            "reduce_time_sec": reduce_time,
                            "total_time_sec": total_pipeline_time,
                            "overall_tokens_per_sec": overall_tps
                        }
                    })

                except Exception as e:
                    print(f"FAILED: {e}")

    
    with open('experiment_2_KG_MR_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\nExperiment 2 complete! Results saved to experiment_2_KG_MR_results.json")

if __name__ == "__main__":
    run_experiment_two_kg_mr(iterations_per_k=50, k_max = 3)