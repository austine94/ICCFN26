#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 05:30:00 2026

Experiment 4: Architectural Ablation (No Map Phase)
Repeats the KG-structured task, but feeds the RAW descriptions directly 
into the Reduce phase. 
*TWEAK*: Now injects the semantic CWE description from the Knowledge Graph 
instead of the literal CWE ID to force semantic reasoning over keyword matching.
"""

import json
import random
import time
import ollama
import re
from llama_index.core import StorageContext, Settings
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.embeddings.ollama import OllamaEmbedding
from cwe2.database import Database

random.seed(100)

def run_experiment_four_ablation(iterations_per_k=5, k_max=10):
    
    # Setup LlamaIndex
    Settings.llm = LlamaIndexOllama(model="llama3.2:3b", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    print("Loading Knowledge Graph...")
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./cve_knowledge_graph")
        graph_store = storage_context.graph_store 
    except Exception as e:
        print(f"Error loading graph: {e}")
        return
    
    print("Loading Dataset...")
    with open('cwe_cve_10k_mapping.json', 'r') as f:
        cve_data = json.load(f)
    
    # Initialize the local MITRE CWE database
    cwe_db = Database()

    models = [
        "llama3.2:3b",
        "gemma3:4b",
        "phi4-mini",
        "llama3:8b",
        "smollm2:360m"
    ]

    all_results = {}

    for k in range(3, k_max+1):
        print(f"\nEvaluating Ablated Architecture for k={k}")
        all_results[str(k)] = []
        
        valid_cwes = {cwe: cves for cwe, cves in cve_data.items() if len(cves) >= k}
        cwe_list = list(valid_cwes.keys())
        
        if not cwe_list:
            print(f"Not enough data for k={k}. Skipping.")
            continue

        for i in range(iterations_per_k):
            target_cwe = random.choice(cwe_list)
            sampled_cve_ids = random.sample(list(valid_cwes[target_cwe].keys()), k)

            try:
                cwe_int = int(target_cwe.replace("CWE-", ""))
                target_weakness = cwe_db.get(cwe_int)
                target_cwe_text = f"{target_weakness.description}"
            except:
                target_cwe_text = "Definition unavailable."
            
            # Fetch RAW CVE descriptions directly from Graph
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

            # Generate Decoys
            available_decoys = [c for c in cwe_list if c != target_cwe]
            decoy_cwes = random.sample(available_decoys, min(5, len(available_decoys)))
            options_list = [target_cwe] + decoy_cwes
            random.shuffle(options_list)
            
            options_text = ""
            for cwe in options_list:
                try:
                    cwe_int = int(cwe.replace("CWE-", ""))
                    weakness = cwe_db.get(cwe_int)
                    options_text += f"- {cwe}: {weakness.name}\n"
                except:
                    options_text += f"- {cwe}: Definition unavailable\n"

            # Evaluate Models
            for model in models:
                print(f"  Testing {model}", end=" ", flush=True)
                
                try:
                    start_time = time.time()
                    
                    # ==========================================
                    # Constructing the monolithic raw prompt
                    # Injecting the description text, not the ID
                    # ==========================================
                    raw_text_blocks = []
                    for cve_id, raw_desc in raw_descriptions:
                        raw_text_blocks.append(
                            f"{cve_id} ({target_cwe_text})\nRaw Description: {raw_desc}"
                        )
                    
                    massive_text_block = "\n\n".join(raw_text_blocks)
                    
                    system_prompt = f"""You are a cyber threat intelligence analyst. 
Identify the common underlying weakness (CWE) shared by these {k} vulnerabilities.
Carefully inspect the prompt as it contains text detailing the underlying weakness from structured knowledge graph lookups.    

Options:
{options_text}

Respond ONLY in strict JSON format matching this schema:
{{"predicted_cwe": "CWE-ID"}}"""

                    user_prompt = f"What is the exact CWE-ID from the options list that matches the common weakness described across these vulnerabilities?\n\n{massive_text_block}"

                    # Single monolithic API call
                    response = ollama.chat(
                        model=model,
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_prompt}
                        ],
                        format='json',
                        options={
                            'temperature': 0.0, 
                            'seed': 100,
                            'num_predict': 50 # Circuit breaker for JSON output
                        }
                    )
                    
                    total_time = time.time() - start_time
                    total_tokens = response.get('eval_count', 1)
                    overall_tps = total_tokens / total_time if total_time > 0 else 0
                    
                    try:
                        prediction = json.loads(response['message']['content'])
                        predicted_cwe = prediction.get('predicted_cwe', 'JSON_PARSE_ERROR')
                    except:
                        predicted_cwe = 'FORMAT_ERROR'

                    is_correct = bool(re.search(rf"\b{target_cwe}\b", predicted_cwe))
                    print(f"Result: {predicted_cwe} (Correct: {is_correct}) | Speed: {overall_tps:.1f} t/s | Time: {total_time:.2f}s")

                    all_results[str(k)].append({
                        "iteration": i+1,
                        "target_cwe": target_cwe,
                        "model": model,
                        "predicted_cwe": predicted_cwe,
                        "is_correct": is_correct,
                        "metrics": {
                            "total_time_sec": total_time,
                            "overall_tokens_per_sec": overall_tps
                        }
                    })

                except Exception as e:
                    print(f"FAILED: {e}")

    with open('experiment_4_ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\nExperiment 4 complete! Results saved to experiment_4_ablation_results.json")

if __name__ == "__main__":
    run_experiment_four_ablation(iterations_per_k=5, k_max=10)