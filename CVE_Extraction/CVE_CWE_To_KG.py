#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:32:49 2026

@author: localadmin
"""

import json
from llama_index.core import StorageContext, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from cwe2.database import Database 

def build_local_graph():
    #initialise models, using Nomic Embed Text
    llm = Ollama(model="llama3.2:3b", request_timeout=120.0)
    embed_model = OllamaEmbedding(model_name="nomic-embed-text") 

    #set up a SimpleGraphStore() for the KG
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    index = KnowledgeGraphIndex.from_documents(
        [], 
        storage_context=storage_context,
        llm=llm,
        embed_model=embed_model
    )

    #need a db of CWE results for evaluation context
    cwe_db = Database()

    #read in CWE/CVE dataset
    with open('cwe_cve_10k_mapping.json', 'r') as f:
        cve_data = json.load(f)

    print("Building the knowledge graph...")

    #keep track of CWEs 
    defined_cwes = set()

    #force together CVEs with the same CWE, as we know this a priori
    for cwe, cves in cve_data.items():
        
        #store CWE official def
        if cwe not in defined_cwes:
            try:
                #extract the integer (e.g., "CWE-79" -> 79) to query the local DB
                cwe_int = int(cwe.replace("CWE-", ""))
                weakness = cwe_db.get(cwe_int)
                
                #combine the official name and description
                cwe_def = f"{weakness.name}: {weakness.description}"
                
                #link the CWE to its definition
                index.upsert_triplet(
                    (cwe, "HAS_DEFINITION", cwe_def)
                )
                defined_cwes.add(cwe)
            except Exception as e:
                print(f"Note: Could not fetch definition for {cwe}: {e}")

        for cve_id, description in cves.items():
            
            #link the CVE to the CWE
            index.upsert_triplet(
                (cve_id, "HAS_WEAKNESS", cwe)
            )
            
            # Link the CVE to its description
            index.upsert_triplet(
                (cve_id, "HAS_DESCRIPTION", description)
            )

    #store KG
    persist_directory = "./cve_knowledge_graph"
    index.storage_context.persist(persist_dir=persist_directory)
    print("Knowledge Graph successfully built and saved!")

if __name__ == "__main__":
    build_local_graph()