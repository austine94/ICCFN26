#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:32:44 2026

@author: localadmin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:14:00 2026

@author: localadmin
"""

import json
import random
import ollama
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set fixed seed for dataset reproducibility
SEED = 100
random.seed(SEED)

#Noise Examples 

SYNTACTIC_NOISE_POOL = [
    " [DEBUG: Stack trace memory dump 0x00A3F1 failed at offset 0x048F] ",
    " &lt;script&gt;alert('XSS_PAYLOAD_TEST')&lt;/script&gt; ",
    " \\n\\t\\t\\t[WARN: Unexpected token in JSON parser at index 1024] ",
    " [ERR_CORRUPTED_PAYLOAD_OFFSET_8812_HEAP_ALLOC] ",
    " %20%27%20OR%201%3D1%20--%20%00 ",
    " *** UNHANDLED_EXCEPTION_ACCESS_VIOLATION_0xC0000005 *** ",
    " { \"error\": \"malformed_unicode_escape_sequence\", \"line\": 42 } ",
    " Segmentation fault (core dumped) at 0x7fffffffdf20 ",
    " &amp;#x3C;img src=x onerror=alert(1)&amp;#x3E; ",
    " \\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00 (binary stream artifact) ",
    " [syslog: kernel BUG at mm/slub.c:3952!] ",
    " FATAL: password authentication failed for user &quot;admin&quot; ",
    " HTTP/1.1 400 Bad Request \\r\\nContent-Length: 0\\r\\n\\r\\n ",
    " java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"str\" is null ",
    " [VULN_SCAN_ARTIFACT: payload=%24%7B%28%23a%3D%40org.apache%29%7D] ",
    " 0xDEADBEEF 0xCAFEBABE 0x00000000 0x00000001 ",
    " \\x90\\x90\\x90\\x90\\x90\\x90\\x90\\x90 (NOP sled artifact) ",
    " TypeError: undefined is not an object (evaluating 'x.target') ",
    " sqlite3.OperationalError: near \"DROP\": syntax error ",
    " Traceback (most recent call last): File \"&lt;stdin&gt;\", line 1, in &lt;module&gt; "
]

BOILERPLATE_NOISE_POOL = [
    " Users are strongly advised to update to version 2.1.4 immediately. See security bulletin MS-441 for mitigation steps and firewall configurations. ",
    " Note: This issue only affects legacy deployments running on unsupported Enterprise Linux 6 kernels. Contact vendor support for backported hotfixes. ",
    " Mitigation: Ensure network access to port 8443 is restricted behind an upstream Web Application Firewall (WAF) and reverse proxy. ",
    " Compliance Notice: CVE severity score updated in accordance with CVSS v3.1 vector specifications. Internal Audit Reference: AUD-2025-99A. ",
    " Enterprise deployments utilizing containerized Kubernetes pods must restart the ingress controller daemon to clear cached session tokens. ",
    " The CVSS v4.0 Base Score is 7.5 (High). The vector string is CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:H/VA:N/SC:N/SI:N/SA:N. ",
    " Customers under active maintenance agreements should apply the cumulative patch bundle released on Patch Tuesday. ",
    " Workaround: Administrators can disable the affected service daemon manually to prevent exploitation until a patch is applied. ",
    " This advisory was published in coordination with the National Vulnerability Database (NVD) and MITRE's CVE Numbering Authority (CNA) program. ",
    " Affected users who cannot immediately upgrade should monitor network traffic for suspicious anomalies using IDS/IPS rules provided in the appendix. ",
    " This vulnerability was privately disclosed to the security response team; we are not aware of any active exploitation in the wild. ",
    " Proof-of-concept (PoC) code is not publicly available at this time, reducing the immediate likelihood of automated exploitation. ",
    " Upgrading to the latest firmware version permanently resolves this vulnerability across all hardware appliance models. ",
    " Cloud-managed instances received automated hotfixes prior to public disclosure; no customer action is required for SaaS environments. ",
    " Administrators must manually regenerate cryptographic keys after applying the patch to ensure complete remediation. ",
    " Refer to Knowledge Base Article KB-89234 for detailed instructions on configuring the registry key workaround. ",
    " The vulnerability severity is rated as Critical due to the low attack complexity and lack of required user interaction. ",
    " Third-party dependencies integrated into the core product are not affected, provided they are updated to their latest stable releases. ",
    " An authorized vulnerability scan reported this finding; however, internal testing confirmed it is a false positive under default configurations. ",
    " To verify successful patching, run the diagnostic script available in the vendor portal under the 'Security Utilities' section. "
]

DECOY_NOISE_POOL = [
    " The system also exhibited potential signs of an SQL injection vulnerability or buffer overflow in the secondary memory heap. ",
    " Secondary analysis indicated cross-site scripting (XSS) vectors and improper privilege escalation in adjacent microservices. ",
    " Discovered during routine automated dynamic testing for unauthenticated remote code execution and path traversal faults. ",
    " Audit reports previously highlighted hardcoded cryptographic keys and insecure deserialization within the same core codebase. ",
    " Exploitation requires authenticated administrative access to trigger memory corruption via crafted command injection parameters. ",
    " While investigating a server-side request forgery (SSRF) flaw, researchers observed anomalous cross-site request forgery (CSRF) tokens. ",
    " The attack chain could theoretically bypass authentication if chained with an XML External Entity (XXE) injection vulnerability. ",
    " Legacy components in the system are known to be susceptible to directory traversal and local file inclusion (LFI) attacks. ",
    " Static analysis flagged a potential use-after-free condition, though dynamic testing could not reliably trigger the memory leak. ",
    " The payload bypasses output sanitization, leading to stored HTML injection and potential arbitrary file upload risks. ",
    " Previous audits discovered open redirect vulnerabilities and missing rate limiting on the same authentication endpoint. ",
    " The threat actor leveraged an unpatched race condition to achieve lateral movement and subsequent denial-of-service (DoS). ",
    " Payload execution mimics an LDAP injection attack, though the backend database is primarily susceptible to NoSQL injection. ",
    " The library was previously deprecated due to severe out-of-bounds write and integer overflow vulnerabilities. ",
    " Security logs indicated an attempted format string vulnerability, but the primary vector relied on OS command injection. ",
    " The mitigation prevents man-in-the-middle (MitM) attacks by enforcing strict TLS, avoiding fallback to insecure plaintext protocols. ",
    " A malicious actor could exploit the concurrent execution thread to trigger a null pointer dereference or infinite loop. ",
    " The configuration lacks anti-clickjacking headers, increasing the risk of UI redressing alongside the primary vulnerability. ",
    " By exploiting the improper access control flaw, an attacker could simulate a reflected XSS attack against administrative users. ",
    " The vulnerable endpoint is also being evaluated for improper authorization and broken object level authorization (BOLA) flaws. "
]

def apply_noise(text, noise_type):
    """
    Injects synthetic real-world noise into a CVE description using
    random selection from designated noise pools.
    """
    if noise_type == "Clean":
        return text
        
    words = text.split()
    if not words:
        return text
        
    # Choose a random insertion index inside the description
    insert_idx = random.randint(0, len(words))
    
    if noise_type == "Syntactic":
        # Select randomly from scraper errors, stack traces, and formatting bugs
        noise = random.choice(SYNTACTIC_NOISE_POOL)
    elif noise_type == "Boilerplate":
        # Select randomly from vendor update advice and compliance notes
        noise = random.choice(BOILERPLATE_NOISE_POOL)
    elif noise_type == "Decoy":
        # Select randomly from misleading vulnerability jargon
        noise = random.choice(DECOY_NOISE_POOL)
    else:
        noise = ""
        
    words.insert(insert_idx, noise)
    return " ".join(words)


def run_tfidf_baseline_noisy(test_samples, noise_type):
    print(f"\n--- Running TF-IDF Baseline | Noise: {noise_type} ---")
    baseline_results = []
    
    for idx, sample in enumerate(test_samples):
        target_cwe = sample["target_cwe"]
        true_indices = sample["true_indices"]
        
        # Apply random noise variant to raw text descriptions
        target_text = apply_noise(sample["target_cve"][1], noise_type)
        candidate_texts = [apply_noise(cand[1], noise_type) for cand in sample["candidates"]]
        
        corpus = [target_text] + candidate_texts
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', analyzer='char_wb')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            target_vec = tfidf_matrix[0]
            candidate_vecs = tfidf_matrix[1:]
            sim_scores = cosine_similarity(target_vec, candidate_vecs)[0]
            
            top_3_zero_based = np.argsort(sim_scores)[::-1][:3]
            predicted_matches = [int(i + 1) for i in top_3_zero_based]
            
        except Exception as e:
            print(f"  TF-IDF Baseline Failed: {e}")
            predicted_matches = []
            
        correct = len(set(predicted_matches).intersection(set(true_indices)))
        accuracy = correct / 3.0
        
        baseline_results.append({
            "model": "tfidf-cosine-baseline",
            "noise_type": noise_type,
            "iteration": idx + 1,
            "accuracy": accuracy
        })
        
    return baseline_results


def run_experiment_one_noise(iterations_per_config=5):
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
    
    noise_types = ["Clean", "Syntactic", "Boilerplate", "Decoy"]
    fixed_temp = 0.0

    valid_cwes = {cwe: cves for cwe, cves in cve_data.items() if len(cves) >= 4}
    cwe_list = list(valid_cwes.keys())
    
    if not cwe_list:
        print("Error: Not enough data.")
        return

    # =========================================================================
    # PRE-GENERATE DATASET SAMPLES
    # =========================================================================
    print(f"Pre-sampling {iterations_per_config} identical test instances...")
    test_samples = []
    for _ in range(iterations_per_config):
        target_cwe = random.choice(cwe_list)
        sampled_cves = random.sample(list(valid_cwes[target_cwe].items()), 4)

        target_cve = sampled_cves[0]
        match_cves = sampled_cves[1:4] 

        decoy_cwes = random.sample([c for c in cwe_list if c != target_cwe], 1)
        decoy_cves = [random.choice(list(valid_cwes[dcwe].items())) for dcwe in decoy_cwes]

        candidates = match_cves + decoy_cves
        random.shuffle(candidates)

        true_indices = [idx + 1 for idx, candidate in enumerate(candidates) if candidate in match_cves]
                
        test_samples.append({
            "target_cwe": target_cwe,
            "target_cve": target_cve,
            "candidates": candidates,
            "true_indices": true_indices
        })

    all_results = []

    # =========================================================================
    # RUN EVALUATIONS ACROSS NOISE TYPES
    # =========================================================================
    for noise in noise_types:
        print(f"\n=========================================")
        print(f"EVALUATING NOISE SCENARIO: {noise}")
        print(f"=========================================")
        
        # 1. Run Non-AI Baseline
        baseline_results = run_tfidf_baseline_noisy(test_samples, noise)
        all_results.extend(baseline_results)

        # 2. Run SLMs
        for model in models:
            print(f"\n--- Testing Model: {model} | Noise: {noise} ---")

            for i, sample in enumerate(test_samples):
                true_indices = sample["true_indices"]

                # Apply random noise variant to raw inputs before mapping
                noisy_target_desc = apply_noise(sample["target_cve"][1], noise)
                noisy_candidates = [(cand_id, apply_noise(desc, noise)) for cand_id, desc in sample["candidates"]]

                map_prompt_template = "Extract the core vulnerability mechanism from this description in 1-2 short sentences.\n\nDescription: {}"

                try:
                    # Map Target
                    target_map_res = ollama.chat(
                        model=model, 
                        messages=[{'role': 'user', 'content': map_prompt_template.format(noisy_target_desc)}], 
                        options={'temperature': fixed_temp, 'num_predict': 150} # Circuit breaker
                    )
                    distilled_target = target_map_res['message']['content'].strip()

                    # Map Candidates
                    distilled_candidates = []
                    for cve_id, desc in noisy_candidates:
                        cand_map_res = ollama.chat(
                            model=model, 
                            messages=[{'role': 'user', 'content': map_prompt_template.format(desc)}], 
                            options={'temperature': fixed_temp, 'num_predict': 150} # Circuit breaker
                        )
                        distilled_candidates.append(cand_map_res['message']['content'].strip())
                except Exception as e:
                    print(f"  FAILED during Map Phase on {model}: {e}")
                    continue

                candidates_text = "\n".join([f"Candidate {idx+1}: {desc}" for idx, desc in enumerate(distilled_candidates)])
                
                system_prompt = """You are a cyber threat intelligence analyst.
Your task is to identify which 3 candidate vulnerabilities share the EXACT SAME underlying technical weakness (CWE) as the target vulnerability. Do not group them by software or product; group them ONLY by the specific vulnerability mechanism.

Respond ONLY in strict JSON format matching this schema:
{
    "target_weakness": "Brief 1-sentence identification of the target's core mechanism.",
    "matches": [integer, integer, integer]
}"""
                
                user_prompt = f"""Target Vulnerability Mechanism:
{distilled_target}

Candidate Mechanisms:
{candidates_text}

Analyze the Target Vulnerability, then provide the numbers of the 3 Candidates that share the exact same underlying weakness in the JSON format."""
                
                try:
                    reduce_response = ollama.chat(
                        model=model,
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_prompt}
                        ],
                        format='json',
                        options={'temperature': fixed_temp, 'num_predict': 150} # Circuit breaker
                    )

                    try:
                        prediction = json.loads(reduce_response['message']['content'])
                        predicted_matches = [int(x) for x in prediction.get('matches', []) if str(x).isdigit()]
                    except:
                        predicted_matches = []

                    correct_predictions = len(set(predicted_matches).intersection(set(true_indices)))
                    accuracy = correct_predictions / 3.0

                    print(f"  Iter {i+1} | True: {sorted(true_indices)} | Pred: {sorted(predicted_matches)} | Acc: {accuracy:.2f}")

                    all_results.append({
                        "model": model,
                        "noise_type": noise,
                        "iteration": i+1,
                        "accuracy": accuracy
                    })

                except Exception as e:
                    print(f"  FAILED during Reduce Phase on {model}: {e}")

    # Save raw output
    output_json = 'experiment_1_noise_results.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=4)

    # =========================================================================
    # GENERATE PIVOT TABLE
    # =========================================================================
    df = pd.DataFrame(all_results)
    summary_df = df.groupby(['model', 'noise_type'])['accuracy'].mean().reset_index()
    summary_df['accuracy'] = summary_df['accuracy'] * 100
    
    pivot_table = summary_df.pivot(index='model', columns='noise_type', values='accuracy')
    
    # Reorder columns to place 'Clean' first
    cols = ['Clean', 'Syntactic', 'Boilerplate', 'Decoy']
    pivot_table = pivot_table[cols]
    
    # Move TF-IDF baseline to the top row
    if 'tfidf-cosine-baseline' in pivot_table.index:
        idx = ['tfidf-cosine-baseline'] + sorted([m for m in pivot_table.index if m != 'tfidf-cosine-baseline'])
        pivot_table = pivot_table.loc[idx]

    print("\n### Experiment 1 Noise: Robustness to Real-World Noise ###")
    print("Values represent: Average Accuracy (%)\n")
    print(pivot_table.to_markdown(floatfmt=".1f"))
    
    output_csv = 'experiment_1_noise_matrix.csv'
    pivot_table.to_csv(output_csv)
    print(f"\nSaved to {output_csv} and {output_json}")

if __name__ == "__main__":
    run_experiment_one_noise(iterations_per_config=100)