#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:26:06 2026

@author: localadmin
"""

import json
import time
import os
import google.generativeai as genai

API_KEY = "" 
if not API_KEY:
    print("ERROR: Please set the GEMINI_API_KEY environment variable.")
    exit()

genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 0.0, #keep the judge deterministic
    "response_mime_type": "application/json", #force strict JSON output
}
model = genai.GenerativeModel(
    model_name="gemini-3.1-flash-lite-preview",
    generation_config=generation_config
)


def run_hallucination_judge():
    try:
        with open('experiment_0_results.json', 'r') as f:
            exp0_data = json.load(f)
    except FileNotFoundError:
        print("Error: experiment_0_results.json not found.")
        return

    hallucination_results = {}
    total_iterations = len(exp0_data)
    
    for iteration, data in exp0_data.items():
        cve_id = data.get("cve_id")
        ground_truth = data.get("ground_truth_description")
        model_responses = data.get("model_responses", {})
        
        print(f"  [Iter {iteration}/{total_iterations}] Checking {cve_id}...", end=" ", flush=True)

        #build a formatted string of the 5 SLM responses
        responses_text = ""
        for slm_name, slm_summary in model_responses.items():
            responses_text += f"\nModel: {slm_name}\nSummary: {slm_summary}\n"

        prompt = f"""You are an expert cybersecurity fact-checker.
            Your only job is to analyse for clear-cut errors in AI-generated summaries of a vulnerability.
            
            A "hallucination" occurs ONLY if the AI summary includes something that is categorically wrong, and is totally unrelated, against the ground truth. Be lenient please, we want clear cut incorrect hallucinated details.
            
            GROUND TRUTH DESCRIPTION:
            "{ground_truth}"
            
            AI SUMMARIES TO EVALUATE:
            {responses_text}
            
            Respond strictly in the following JSON format:
            {{
                "evaluations": [
                    {{
                        "model": "model_name",
                        "has_hallucination": boolean,
                        "hallucination_details": "If true, quote or describe the exact fabricated detail. If false, write 'None'."
                    }},
                    ...
                ]
            }}"""

        try:
            response = model.generate_content(prompt)
            evaluation_json = json.loads(response.text)
            
            hallucination_results[iteration] = {
                "cve_id": cve_id,
                "hallucination_checks": evaluation_json.get("evaluations", [])
            }
            print("Done.")
            
        except Exception as e:
            print(f"FAILED. Error: {e}")
            hallucination_results[iteration] = {
                "cve_id": cve_id,
                "error": str(e)
            }

        #rate limiting: 4 seconds between calls
        time.sleep(4)

    with open('experiment_0_hallucinations.json', 'w') as f:
        json.dump(hallucination_results, f, indent=4)
        
    print("\nDetection complete! Results saved to experiment_0_hallucinations.json")

if __name__ == "__main__":
    run_hallucination_judge()
