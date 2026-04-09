#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:18:37 2026

@author: localadmin
"""

import json
import time
import os
import google.generativeai as genai


API_KEY = "AIzaSyCihTAKvX-PJpGJgBINeIQQBf1jdfSHieY" 
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

def run_gemini_judge():
    try:
        with open('experiment_0_results.json', 'r') as f:
            exp0_data = json.load(f)
    except FileNotFoundError:
        print("Error: experiment_0_results.json not found.")
        return

    evaluations = {}
    total_iterations = len(exp0_data)
    
    for iteration, data in exp0_data.items():
        cve_id = data.get("cve_id")
        ground_truth = data.get("ground_truth_description")
        model_responses = data.get("model_responses", {})
        
        print(f"  [Iter {iteration}/{total_iterations}] Grading {cve_id}...", end=" ", flush=True)

        #build a formatted string of the 5 SLM responses
        responses_text = ""
        for slm_name, slm_summary in model_responses.items():
            responses_text += f"\nModel: {slm_name}\nSummary: {slm_summary}\n"

        prompt = f"""You are an expert cybersecurity peer-reviewer.
            Your task is to evaluate 5 different AI-generated summaries of a vulnerability.
            
            GROUND TRUTH DESCRIPTION:
            "{ground_truth}"
            
            AI SUMMARIES TO EVALUATE:
            {responses_text}
            
            EVALUATION CRITERIA:
            For each model, provide a score from 1 to 10 based on how accurately and concisely it captured the core vulnerability mechanism and impact from the ground truth.
            
            Respond strictly in the following JSON format:
            {{
                "evaluations": [
                    {{
                        "model": "model_name",
                        "score": integer,
                        "rationale": "1-sentence explanation of the score"
                    }},
                    ...
                ]
            }}"""

        try:
            #call the Gemini API
            response = model.generate_content(prompt)
            
            #the output is guaranteed to be a JSON string due to response_mime_type
            evaluation_json = json.loads(response.text)
            evaluations[iteration] = {
                "cve_id": cve_id,
                "gemini_evaluations": evaluation_json.get("evaluations", [])
            }
            print("Done.")
            
        except Exception as e:
            print(f"FAILED. Error: {e}")
            evaluations[iteration] = {
                "cve_id": cve_id,
                "error": str(e)
            }

        #rate limiting: 4 seconds between calls ensures we stay under 15 RPM
        time.sleep(4)

    # Save the final graded results
    with open('experiment_0_evaluations.json', 'w') as f:
        json.dump(evaluations, f, indent=4)
        
    print("\nEvaluation complete! Results saved to experiment_0_evaluations.json")

if __name__ == "__main__":
    run_gemini_judge()