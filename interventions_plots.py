import os
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=8)

folder = "figures/interventions"
if not os.path.exists(folder):
    os.makedirs(folder, exist_ok=True)

TASKS = ["ioi", "gp",  "gt"]
MODELS = ["gpt2-small", "pythia-160m", "gemma-2-2b"]

logit_diff_ablations = pd.DataFrame()

for task in TASKS:
    files = glob.glob(f"intervention_data/*_{task}.parquet")

    for file in files:
        df_file = pd.read_parquet(file)
        df_file["task"] = task
        logit_diff_ablations = pd.concat([
            logit_diff_ablations,
            df_file
        ]).reset_index(drop=True)

logit_diff_ablations["edge_labeled_group"] = logit_diff_ablations["edge_labeled_group"].apply(lambda x: "->\n".join(x.split("-> ")))

fig, ax = plt.subplots(len(TASKS), len(MODELS), sharex=False, sharey=False, figsize=(5.5, 4.4))
fig.delaxes(ax[2, 2])
for i, task in enumerate(TASKS):
    for j, model in enumerate(MODELS): 
        data = logit_diff_ablations[
            (logit_diff_ablations.is_ablated) & 
            (logit_diff_ablations.intervention_type == "local") &
            (logit_diff_ablations.model_name == model) &
            (logit_diff_ablations.task == task)
        ]
        sns.violinplot(
            data,
            x="logit_diff_metric",
            y="edge_labeled_group",
            hue="operation_performed", 
            hue_order=['Removing (Random)', 'Boosting (Random)', 'Removing (SVs)', 'Boosting (SVs)'], 
            linewidth=0.5,
            legend=True if i == 2 and j == 1 else False,
            density_norm = 'width',
            inner = None,
            cut=0,
            ax=ax[i, j],
        )

        ax[i, j].set_xlabel("(F(E, h) - F) / F");
        if j == 0:
            ax[i, j].set_ylabel(f"{task.upper()}");
        else:
            ax[i, j].set_ylabel("");

        if j > 0:
            ax[i, j].set_yticklabels([])

        if i != len(TASKS) - 1:
            ax[i, j].set_xlabel("")

        ax[i, j].axvline(0, color='black', linestyle='--', linewidth=0.5)

        if i == 0:
            ax[i, j].set_title(model)

        if model == "pythia-160m" and task == "gt":
            ax[i, j].set_xlim(-3, 3)

#plt.tight_layout()
plt.legend(loc="center", bbox_to_anchor=(1.75, 0.5), fontsize=6);
plt.savefig(f'figures/interventions/interventions.pdf', bbox_inches='tight', dpi=800);
plt.close();

# GPT-2 small only
from copy import deepcopy

fig, ax = plt.subplots(1, len(TASKS), sharex=False, sharey=True, figsize=(6.2, 1.5))
model = "gpt2-small"
for i, task in enumerate(TASKS):
    data = deepcopy(logit_diff_ablations[
        (logit_diff_ablations.is_ablated) & 
        (logit_diff_ablations.intervention_type == "local") &
        (logit_diff_ablations.model_name == model) &
        (logit_diff_ablations.task == task)
    ])

    data["edge_labeled_group"].replace({
        'S-Inhibition Head ->\nName Mover Head': 'S-Inhibition ->\nName Mover',
        'Induction Head ->\nS-Inhibition Head': 'Induction ->\nS-Inhibition',
        'Previous Token Head ->\nInduction Head': 'Prev. Token ->\nInduction'
    }, inplace=True)    

    sns.violinplot(
        data,
        x="edge_labeled_group",
        y="logit_diff_metric",
        hue="operation_performed", 
        hue_order=['Removing (Random)', 'Boosting (Random)', 'Removing (SVs)', 'Boosting (SVs)'], 
        linewidth=0.5,
        #legend=True if i == 1 else False,
        legend=False,
        density_norm = 'width',
        inner = None,
        cut=0,
        ax=ax[i],
    )

    ax[i].set_ylabel("(F(E, h) - F) / F");
    ax[i].set_xlabel("")
    ax[i].set_yticks([-0.5, 0, 0.5])
    ax[i].set_ylim(-0.5, 0.5)
    ax[i].set_xticklabels(ax[i].get_xticklabels() ,rotation=60)

    # if j > 0:
    #     ax[i].set_yticklabels([])

    # if i != len(TASKS) - 1:
    #     ax[i].set_xlabel("")

    ax[i].axhline(0, color='black', linestyle='--', linewidth=0.5)

    
    # ax[i].set_title(task.upper(), fontsize=8)

    # if model == "pythia-160m" and task == "gt":
    #     ax[i].set_xlim(-3, 3)

#plt.tight_layout()
#ax[1].legend(loc='upper center', bbox_to_anchor=(1, 3), ncol=2, fontsize=6);

plt.savefig(f'figures/interventions/interventions_{model}.pdf', bbox_inches='tight', dpi=800);
plt.close();

fig, ax = plt.subplots(len(TASKS), len(MODELS), sharex=False, sharey=True, figsize=(5.5, 4.4))
fig.delaxes(ax[2, 2])
for i, task in enumerate(TASKS):
    for j, model in enumerate(MODELS): 
        data = logit_diff_ablations[
            (logit_diff_ablations.is_ablated) & 
            (logit_diff_ablations.intervention_type == "local") &
            (logit_diff_ablations.model_name == model) &
            (logit_diff_ablations.task == task)
        ]
        
        sns.barplot(
            data,
            x="edge_labeled_group",
            y="scores_dest_src_diff_metric",
            hue="operation_performed", 
            hue_order=['Removing (Random)', 'Boosting (Random)', 'Removing (SVs)', 'Boosting (SVs)'], 
            legend=True if i == 2 and j == 1 else False,
            errorbar="sd",
            ax=ax[i, j],
            err_kws={"linewidth": 1},
        )

        ax[i, j].set_xlabel(None)
        if j == 0:
            ax[i, j].set_ylabel(f"{task.upper()}\n" + r"$A_{ds}^{\text{interv}} - A_{ds}$")

        ax[i, j].set_xticklabels([])
        if i == 2:
            ax[i, j].set_xlabel("Edges")

        if i == 0:
            ax[i, j].set_title(model)

        if i == 2 and j == 1:
            plt.legend(loc="center", bbox_to_anchor=(1.75, 0.5), fontsize=6)

plt.savefig(f'figures/interventions/interventions_attn_weight_effect.pdf', bbox_inches='tight', dpi=800);
plt.close();

for metric in ["cosine_similarity", "norm_ratio"]:
    for is_random in [True, False]:
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.9))
        sns.histplot(
            logit_diff_ablations[(logit_diff_ablations.is_ablated) & (logit_diff_ablations.is_random == is_random)],
            x=metric,
            ax=ax
        );
        if metric == "cosine_similarity": 
            plt.xlabel("Cosine similarity");
        else:
            plt.xlabel("Norm ratio");
        random_name = "random" if is_random else "not-random"
        plt.tight_layout();
        plt.savefig(f'figures/interventions/interventions_{metric}_{random_name}.pdf', bbox_inches='tight', dpi=800);
        plt.close();