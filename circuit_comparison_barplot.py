import os
import pandas as pd

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import glob

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=8)

csv_files = glob.glob("circuit_comparison/*.csv")

df = pd.DataFrame()
for file in csv_files:
    df = pd.concat([df, pd.read_csv(file)]).reset_index(drop=True)
df["Method B"].replace({"Ours w/ th=0.2": "Ours", "Edge Pruning": "EP"}, inplace=True)

fig, ax = plt.subplots(1, 4, sharey=True, sharex=False, figsize=(6.8, 2))

i = 0
for task in ["ioi", "gp", "gt"]:
    if task == "ioi":
        for model in df.Model.unique():
            sns.barplot(df[(df.Task == task) & (df.Model == model)], x="Method B", y="Value", hue="Metric", ax=ax[i], legend=False)
            ax[i].set_title(f"{model}, {task.upper()}")
            ax[i].set_xlabel(None)
            ax[i].set_ylabel("Metric Value")
            i += 1
    else:
        sns.barplot(df[(df.Task == task) & (df.Model == model)], x="Method B", y="Value", hue="Metric", ax=ax[i], legend=False if i < 3 else True)
        ax[i].set_title(f"Model: {model}, Task: {task.upper()}")
        ax[i].set_xlabel(None)
        ax[i].set_title(f"{model}, {task.upper()}")
        i += 1
plt.legend(loc='lower center', bbox_to_anchor=(-1.2, -0.3), ncol=3);
plt.tight_layout()
folder = f"figures/circuit_comparison"
if not os.path.exists(folder):
    os.makedirs(folder, exist_ok=True)
plt.savefig(f'{folder}/circuit_comparison.pdf', bbox_inches='tight', dpi=800);
plt.close();