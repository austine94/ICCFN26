#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:23:36 2026

@author: localadmin
"""

import json
import requests
import time

def fetch_cves_by_cwe():
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    cwe_mapping = {}
    results_per_page = 200 
    target_total = 10000
    
    for start_index in range(0, target_total, results_per_page):
        params = {
            'resultsPerPage': results_per_page,
            'startIndex': start_index
        }
        
        print(f"Fetching vulnerabilities {start_index} to {start_index + results_per_page}...")
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}. The NVD API might be throttling you.")
            break
            
        data = response.json()
        vulnerabilities = data.get('vulnerabilities', [])
        
        for item in vulnerabilities:
            cve_data = item.get('cve', {})
            cve_id = cve_data.get('id')
            
            #extract description
            descriptions = cve_data.get('descriptions', [])
            cve_desc = next((d['value'] for d in descriptions if d['lang'] == 'en'), "No description.")
            
            #extract CWEs
            weaknesses = cve_data.get('weaknesses', [])
            for weakness in weaknesses:
                descriptions = weakness.get('description', [])
                for desc in descriptions:
                    cwe_id = desc.get('value')
                    
                    if cwe_id and cwe_id.startswith("CWE-"):
                        if cwe_id not in cwe_mapping:
                            cwe_mapping[cwe_id] = {}
                        cwe_mapping[cwe_id][cve_id] = cve_desc

        #sleep to respect the 5 requests / 30 seconds rate limit (without API key)
        print("Sleeping for 7 seconds to avoid rate limits...")
        time.sleep(7) 

    return cwe_mapping

if __name__ == "__main__":
    dataset = fetch_cves_by_cwe()
    
    #filter out CWEs with too few CVEs to ensure good evaluation data
    #so only keep CWEs that have at least 3 CVEs attached to them
    filtered_dataset = {k: v for k, v in dataset.items() if len(v) >= 3}
    
    with open('cwe_cve_10k_mapping.json', 'w') as f:
        json.dump(filtered_dataset, f, indent=4)
        
    print(f"Dataset generated with {len(filtered_dataset)} unique CWE categories!")