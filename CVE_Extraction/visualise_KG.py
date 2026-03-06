import json
import networkx as nx
from pyvis.network import Network

def visualize_graph_direct(cve_limit = 500):
    with open('cwe_cve_10k_mapping.json', 'r') as f:
        cve_data = json.load(f)

    #create a NetworkX directed graph
    nx_graph = nx.DiGraph()

    #limit the number of CVEs rendered so your browser doesn't freeze!
    cve_limit = cve_limit
    cve_count = 0

    print("Populating the graph...")
    #build the nodes and edges directly
    for cwe, cves in cve_data.items():
        #add the CWE node 
        nx_graph.add_node(cwe, size=30, color="#d9534f", title="CWE Vulnerability Type")
        
        for cve_id, description in cves.items():
            if cve_count >= cve_limit:
                break
            
            #add the CVE node (Smaller and Blue). The title shows up on hover!
            nx_graph.add_node(cve_id, size=15, color="#5bc0de", title=description)
            
            #create the edge connecting them
            nx_graph.add_edge(cve_id, cwe, label="HAS_WEAKNESS")
            
            cve_count += 1
            
        if cve_count >= cve_limit:
            break

    print(f"Graph built with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges.")

    #start up PyVis Network
    print("Building HTML visualization...")
    net = Network(
        height="800px", 
        width="100%", 
        directed=True, 
        bgcolor="#ffffff", 
        font_color="black"
    )
    
    net.from_nx(nx_graph)
    net.show_buttons(filter_=['physics'])
    
    output_file = "cve_KG_visualisation.html"
    net.save_graph(output_file)

if __name__ == "__main__":
    visualize_graph_direct()