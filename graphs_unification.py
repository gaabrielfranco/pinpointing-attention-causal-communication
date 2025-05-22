import argparse
import glob
import os
from pprint import pprint
import networkx as nx
import numpy as np

def main(args):
    graphs_path = glob.glob(f"traced_graphs/{args.model_name}/{args.task}/*.graphml")
    
    # Sanity check
    if args.task == "ioi" and args.model_name == "gpt2-small":
        n_prompts = 256
        assert len(graphs_path) == 230 # 256 - 25 prompts that do not predict IO
    elif args.task == "ioi" and args.model_name == "pythia-160m":
        n_prompts = 256
        assert len(graphs_path) == 159 # 256 - 97 prompts that do not predict IO
    elif args.task == "ioi" and args.model_name == "gemma-2-2b":
        n_prompts = 256
        assert len(graphs_path) == 206
    elif args.task == "gt" and args.model_name == "gpt2-small":
        n_prompts = 256
        assert len(graphs_path) == 166 # Tracing only prompts with AH writing in the logits
    elif args.task == "gt" and args.model_name == "pythia-160m":
        n_prompts = 256
        assert len(graphs_path) == 39 # Tracing only prompts with AH writing in the logits
    elif args.task == "gp" and args.model_name == "gpt2-small":
        n_prompts = 100
        assert len(graphs_path) == 100
    elif args.task == "gp" and args.model_name == "pythia-160m":
        n_prompts = 100
        assert len(graphs_path) == 99
    elif args.task == "gp" and args.model_name == "gemma-2-2b":
        n_prompts = 100
        assert len(graphs_path) == 94

    # Gathering all the graphs
    G = [None for _ in range(n_prompts)]
    for g_path in graphs_path:
        prompt_id = eval(g_path.split("/")[-1].split("_")[-2])
        G[prompt_id] = nx.read_graphml(g_path)

    # Building the combined graph
    G_combined = nx.MultiDiGraph()
    for prompt_id, G_curr in enumerate(G):
        if G_curr is None:
            continue
        
        for edge in G_curr.edges(data=True):
            if (edge[0], edge[1]) in G_combined.edges:
                # Updating edges (weight)
                if len(G_combined[edge[0]][edge[1]]) > 2:
                    raise Exception("Bug!")
                elif len(G_combined[edge[0]][edge[1]]) == 2:
                    # Multiedge with type=d and type=s. Getting the first one
                    candidate_edge = G_combined[edge[0]][edge[1]][0]

                    # Updating the weight of the correct edge
                    if candidate_edge["type"] == edge[2]["type"]:
                        G_combined.edges[(edge[0], edge[1], 0)]["weight"] += edge[2]["weight"]
                        G_combined.edges[(edge[0], edge[1], 0)]["prompts_appeared"].append(prompt_id)
                    else:
                        G_combined.edges[(edge[0], edge[1], 1)]["weight"] += edge[2]["weight"]
                        G_combined.edges[(edge[0], edge[1], 1)]["prompts_appeared"].append(prompt_id)
                else:                    
                    # If they are the same type, we just add
                    if G_combined.edges[(edge[0], edge[1], 0)]["type"] == edge[2]["type"]:
                        G_combined.edges[(edge[0], edge[1], 0)]["weight"] += edge[2]["weight"]
                        G_combined.edges[(edge[0], edge[1], 0)]["prompts_appeared"].append(prompt_id)
                    else: # If they are not the same type, we create a new edge
                        G_combined.add_edge(
                            edge[0],
                            edge[1],
                            prompts_appeared=[prompt_id],
                            **edge[2],
                        )
            else:                    
                G_combined.add_edge(
                    edge[0],
                    edge[1],
                    prompts_appeared=[prompt_id],
                    **edge[2],
                )

        for node in G_curr.nodes:
            # Updating nodes (n_appearences)
            if "n_appearences" in G_combined.nodes[node]:
                G_combined.nodes[node]["n_appearences"] += 1
            else:
                G_combined.nodes[node]["n_appearences"] = 1
        
    n_prompts_appeared_edges = []
    for _, _, data in G_combined.edges(data=True):
        n_prompts_appeared_edges.append(len(data["prompts_appeared"]))

    assert np.max(n_prompts_appeared_edges) <= len(graphs_path)
    
    # Normalizing the edge weight
    for edge in G_combined.edges:
        G_combined.edges[edge]["weight"] /= len(graphs_path)
        # Converting list to str to be saved by graphML
        G_combined.edges[edge]["prompts_appeared"] = str(G_combined.edges[edge]["prompts_appeared"])


    # Prunning the combined graph using multiple thresholds and save them
    base_name = "_".join(graphs_path[0].split("/")[-1].split("_")[:-2])

    # Creating the combined graph folder
    combined_graph_folder = f"combined_graphs/{args.model_name.split('/')[-1]}/{args.task}"
    if not os.path.exists(combined_graph_folder):
        os.makedirs(combined_graph_folder, exist_ok=True)

    n_appearences = []
    for node in G_combined.nodes(data=True):
        n_appearences.append(node[1]["n_appearences"])

    print(f"Combined graph with {len(G_combined.nodes())} nodes and {len(G_combined.edges())} edges")

    # Saving the combined graph with no threshold
    nx.write_graphml(G_combined, f"{combined_graph_folder}/{base_name}_combined_0.0.graphml")

    for thresh in [0.01, 0.05] + list(np.arange(0.1, 1.0, 0.1)):
        n_appearences_threshold = int(round(thresh * len(graphs_path), 0))
        G_combined_prunned = nx.MultiDiGraph(nx.subgraph_view(G_combined, filter_node=lambda node: G_combined.nodes[node]["n_appearences"] >= n_appearences_threshold))

        # Removing degree zero nodes
        G_combined_prunned.remove_nodes_from([node for node,degree in dict(G_combined_prunned.degree()).items() if degree == 0])
        
        if len([len(x) for x in nx.weakly_connected_components(G_combined_prunned)]) > 1:
            print("There is more than one connected component in the graph. Check it out!")
            
        # Saving
        nx.write_graphml(G_combined_prunned, f"{combined_graph_folder}/{base_name}_combined_{np.round(thresh, 2)}.graphml")
        print(f"Prunned graph (th={np.round(thresh, 2)}, n>={n_appearences_threshold}) with {len(G_combined_prunned.nodes())} nodes and {len(G_combined_prunned.edges())} edges")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", choices=["gpt2-small", "pythia-160m", "gemma-2-2b"], required=True)
    parser.add_argument("-t", "--task", choices=["ioi", "gt", "gp"], required=True)
    args = parser.parse_args()

    main(args)
