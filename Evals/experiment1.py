
import json
import random
import ollama

random.seed(100)

def run_experiment_one(iterations_per_config=5):
    '''
    This tests how well the models can, using information from the knowledge
    graph, determine which 3 CVEs out of a possible 5 are connected to a given
    target CVE.
    
    We study this for varying levels of temperature.
    '''
    print("Loading Dataset...")
    try:
        with open('cwe_cve_10k_mapping.json', 'r') as f:
            cve_data = json.load(f)
    except FileNotFoundError:
        print("Error: cwe_cve_10k_mapping.json not found.")
        return

    #models and Temperature Matrix
    models = [
        "llama3.2:3b",
        "gemma3:4b",
        "phi4-mini",
        "llama3:8b",
        "smollm2:360m"
    ]
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    #need CWEs that have at least 4 CVEs (1 Target + 3 Matches)
    valid_cwes = {cwe: cves for cwe, cves in cve_data.items() if len(cves) >= 4}
    cwe_list = list(valid_cwes.keys())
    
    if not cwe_list:
        print("Error: Not enough data. No CWE has >= 4 CVEs.")
        return

    all_results = []
    print("Starting Experiment 1 & 1a: Reasoning and Temperature Matrix...")

    for model in models:
        for temp in temperatures:

            for i in range(iterations_per_config):
                # Pick a target CWE and get 4 CVEs (1 target, 3 matches)
                target_cwe = random.choice(cwe_list)
                sampled_cves = random.sample(list(valid_cwes[target_cwe].items()), 4)

                target_cve = sampled_cves[0]
                match_cves = sampled_cves[1:4] 

                # Pick 2 decoy CWEs and sample 1 CVE from each
                decoy_cwes = random.sample([c for c in cwe_list if c != target_cwe], 2)
                decoy_cves = []
                for dcwe in decoy_cwes:
                    dcve = random.choice(list(valid_cwes[dcwe].items()))
                    decoy_cves.append(dcve)

                #combine the 3 matches and 2 decoys, then shuffle
                candidates = match_cves + decoy_cves
                random.shuffle(candidates)

                #find the 1-based list indices of the true matches for scoring
                true_indices = []
                for idx, candidate in enumerate(candidates):
                    if candidate in match_cves:
                        true_indices.append(idx + 1)

                candidates_text = "\n".join([f"Candidate {idx+1}: {desc}" for idx, (cve_id, desc) in enumerate(candidates)])
                system_prompt = """You are a cyber threat intelligence analyst.
                Your task is to identify which 3 candidate vulnerabilities share the EXACT SAME underlying technical weakness (CWE) as the target vulnerability. Do not group them by software or product; group them ONLY by the specific vulnerability mechanism (e.g., Buffer Overflow, SQL Injection).
                
                Respond ONLY in strict JSON format matching this schema:
                {
                    "target_weakness": "Brief 1-sentence identification of the target's core mechanism.",
                    "matches": [integer, integer, integer]
                }
                
                EXAMPLE:
                Target: "A flaw in AppX login allows remote attackers to inject arbitrary SQL commands."
                Candidates: 
                1: "Buffer overflow in the memory parser..."
                2: "Improper sanitization of user input in the DB query allows SQL injection..."
                3: "Cross-site scripting (XSS) in the admin panel..."
                4: "SQL injection vulnerability in the reporting module..."
                5: "Failure to escape special characters in SQL commands..."
                
                Correct Output:
                {
                    "target_weakness": "SQL Injection vulnerability due to improper input sanitization.",
                    "matches": [2, 4, 5]
                }
                """
                
                user_prompt = f"""Target Vulnerability:
                {target_cve[1]}
                
                Candidates:
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
                        options={'temperature': temp,
                                 'num_predict': 50}
                    )

                    #evaluation step
                    try:
                        prediction = json.loads(response['message']['content'])
                        predicted_matches = prediction.get('matches', [])
                        
                        # Failsafe: ensure everything in the list is cast to an integer
                        predicted_matches = [int(x) for x in predicted_matches if str(x).isdigit()]
                    except:
                        predicted_matches = []

                    #calculate fractional score (0.0 to 1.0) using set intersection
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
                    print(f"  FAILED on {model} at Temp {temp}: {e}")

    with open('experiment_1_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    print("\nExperiment 1 & 1a complete! Results saved to experiment_1_results.json")

if __name__ == "__main__":
    run_experiment_one(iterations_per_config=200)