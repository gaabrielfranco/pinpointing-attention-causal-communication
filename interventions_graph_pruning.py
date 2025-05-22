import argparse
import os
import networkx as nx
import numpy as np

def main(model_name: str, task: str, n_prompts: int) -> None:
    if model_name == "pythia-160m" and task == "ioi":
        print(f"{model_name}, {task}")
        ROOT_NODE = "('IO-S direction', 'end')" # IOI case
        G: nx.MultiDiGraph = nx.read_graphml("interventions_graph_pythia-160m_ioi_n256_0_combined_0.01.graphml")
        n_prompts_thresh = 15
    elif task == "ioi":
        print(f"{model_name}, {task}")
        ROOT_NODE = "('IO-S direction', 'end')" # IOI case
        G: nx.MultiDiGraph = nx.read_graphml("interventions_graph_gpt2-small_ioi_n256_0_combined_0.01.graphml")
        n_prompts_thresh = 30
    elif task == "gp":
        print(f"{model_name}, {task}")
        ROOT_NODE = "('Correct - Incorrect pronoum', 'end')"
        G: nx.MultiDiGraph = nx.read_graphml("interventions_graph_gpt2-small_gp_n100_0_combined_0.01.graphml")
        n_prompts_thresh = 15
    elif task == "gt":
        print(f"{model_name}, {task}")
        ROOT_NODE = "('True YY - False YY', 'end')"
        G: nx.MultiDiGraph = nx.read_graphml("interventions_graph_gpt2-small_gt_n256_0_combined_0.01.graphml")
        n_prompts_thresh = 15
    print(G)

    for edge in G.edges(keys=True):
        G.edges[edge]["prompts_appeared"] = eval(G.edges[edge]["prompts_appeared"])
        if not ROOT_NODE in edge:
            G.edges[edge]["upstream_node"] = eval(G.edges[edge]["upstream_node"])
            G.edges[edge]["downstream_node"] = eval(G.edges[edge]["downstream_node"])
            G.edges[edge]["logit_diff"] = eval(G.edges[edge]["logit_diff"])
    
    for th in np.arange(0, 1.05, 0.05):
        edges_keep = []
        for edge in G.edges(keys=True):
            if ROOT_NODE in edge:
                edges_keep.append(edge)
                continue
            
            if len(G.edges[edge]["prompts_appeared"]) >= n_prompts_thresh:
                if np.abs(np.mean(G.edges[edge]["logit_diff"])) >= th:
                    edges_keep.append(edge)

        G_pruned = nx.MultiDiGraph(nx.subgraph_view(G, filter_edge=lambda x,y,z: (x,y,z) in edges_keep))
        # Removing degree zero nodes
        G_pruned.remove_nodes_from([node for node,degree in dict(G_pruned.degree()).items() if degree == 0])
        print(G_pruned)

        # Removing the CCs that do not have the ROOT_NODE (logit node)
        for cc in list(nx.weakly_connected_components(G_pruned)):
            if ROOT_NODE in cc:
                continue
            else:
                # remove
                for node in cc:
                    G_pruned.remove_node(node)
        print(G_pruned)

        for edge in G_pruned.edges(keys=True):
            G_pruned.edges[edge]["prompts_appeared"] = str(G_pruned.edges[edge]["prompts_appeared"])
            if not ROOT_NODE in edge:
                try:
                    G_pruned.edges[edge]["upstream_node"] = str(G_pruned.edges[edge]["upstream_node"])
                    G_pruned.edges[edge]["downstream_node"] = str(G_pruned.edges[edge]["downstream_node"])
                    G_pruned.edges[edge]["logit_diff"] = str(G_pruned.edges[edge]["logit_diff"])
                except:
                    print(edge)
                    print(G_pruned.edges[edge])
                    break

        # Normalizing the weights
        for node in G_pruned.nodes():
            total_contribution = {"d": 0, "s": 0}
            for edge in G_pruned.in_edges(node, data=True):
                total_contribution[edge[2]["type"]] += edge[2]["weight"]

            for edge in G_pruned.in_edges(node, keys=True):
                G_pruned.edges[edge]["norm_weight"] = G_pruned.edges[edge]["weight"] / total_contribution[G_pruned.edges[edge]["type"]]

        # Adding abs(avg(logit_diff))
        for edge in G_pruned.edges(keys=True):
            if not ROOT_NODE in edge:
                G_pruned.edges[edge]["abs_avg_logit_diff"] = np.abs(np.mean(eval(G_pruned.edges[edge]["logit_diff"])))
            else:
                G_pruned.edges[edge]["abs_avg_logit_diff"] = 2. # Large numbe to show up

        graph_folder = f"combined_graphs_intervention/{model_name}/{task}"
        if not os.path.exists(graph_folder):
            os.makedirs(graph_folder, exist_ok=True)

        print(f"Saving: {graph_folder}/{model_name}_{task}_n{n_prompts}_combined_{np.round(th, 2)}.graphml")
        print()
        nx.write_graphml(G_pruned, f"{graph_folder}/{model_name}_{task}_n{n_prompts}_combined_{np.round(th, 2)}.graphml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", choices=["gpt2-small", "pythia-160m"], required=True)
    parser.add_argument("-t", "--task", choices=["ioi", "gt", "gp"], required=True)
    parser.add_argument("-n", "--num_prompts", type=int, required=True)
    args = parser.parse_args()

    main(args.model_name, args.task, args.num_prompts)
