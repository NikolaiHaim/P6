import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# IEEE-style plotting defaults
# -----------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,

    "axes.linewidth": 0.8,

    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,

    "lines.linewidth": 1.5,
    "lines.markersize": 4,

    "figure.dpi": 300,
    "savefig.dpi": 600,

    "savefig.bbox": "tight"
})

INPUT_FILE = "ablation.json"
OUTPUT_DIR = Path("plots")

METRICS = [
    "AUC",
    "AP",
    "MRR",
    "Hit@50",
    "Recall@50",
    "NDCG@50",
    "Mean rank",
    "Median rank",
    "Training time"
]

# Reference/baseline runs for each model
REFERENCE_RUNS = {
    "gcn": "baseline",
    "n2v": "baseline",
    "svd": "128_dim",
    "specemb": "128_dim"
}


# -----------------------------------------
# Load + flatten
# -----------------------------------------

def load_ablation(path):

    with open(path) as f:
        return json.load(f)


def flatten_ablation(data):

    rows = []

    for model, experiments in data.items():

        for hyperparam, metrics in experiments.items():

            row = {
                "Model": model,
                "Hyperparameter": hyperparam
            }

            row.update(metrics)

            rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------
# Plotting
# -----------------------------------------

def plot_metric(df, metric):

    models = sorted(df["Model"].unique())

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.4)  # IEEE-ish width
    )

    axes = axes.flatten()

    handles = []
    labels = []


    for ax, model in zip(axes, models):

        model_df = (
            df[df["Model"] == model]
            .copy()
        )

        try:
            model_df["Hyperparameter"] = pd.to_numeric(
                model_df["Hyperparameter"]
            )

            model_df = model_df.sort_values(
                "Hyperparameter"
            )

        except:
            pass


        x = model_df["Hyperparameter"]
        y = model_df[metric]

        # ---------------------------------
        # Reference baseline line
        # ---------------------------------

        ref_key = REFERENCE_RUNS.get(model)

        if ref_key is not None:

            ref_row = model_df[
                model_df["Hyperparameter"].astype(str) == ref_key
            ]

            if not ref_row.empty:

                ref_value = ref_row.iloc[0][metric]

                if pd.notna(ref_value):

                    ax.axhline(
                        y=ref_value,
                        color="0.7",           # light gray
                        linewidth=0.7,         # thin
                        linestyle=(0, (10, 8)), # long sparse dashes
                        alpha=0.7,
                        zorder=0               # behind data
                    )

        line = ax.plot(
            x,
            y,
            color="black",
            linestyle="--",
            marker="o",
            markerfacecolor="black",
            markeredgecolor="black",
            linewidth=1.5,
            markersize=4,
            label=model
        )

        if not handles:
            handles, labels = (
                ax.get_legend_handles_labels()
            )


        ax.set_title(
            model.upper(),
            pad=4
        )

        ax.grid(True)


        # rotate x labels
        ax.tick_params(
            axis='x',
            labelrotation=30
        )


        # align labels nicely
        for label in ax.get_xticklabels():
            label.set_horizontalalignment(
                "right"
            )


        # dynamic y range

        y_min = np.min(y)
        y_max = np.max(y)

        y_range = y_max - y_min

        if y_range == 0:
            y_range = max(
                abs(y_max * 0.05),
                0.01
            )

        padding = y_range * 0.1

        ax.set_ylim(
            y_min - padding,
            y_max + padding
        )


        # lower rank = better
        if metric in [
            "Mean rank",
            "Median rank"
        ]:
            ax.invert_yaxis()


    for i in range(
        len(models),
        len(axes)
    ):
        axes[i].axis("off")

    fig.suptitle(
        metric,
        fontsize=11,
        y=0.98
    )

    plt.subplots_adjust(
        hspace=0.4,
        wspace=0.3
    )

    plt.savefig(
        OUTPUT_DIR /
        f"{metric}.png"
    )

    plt.close()

# -----------------------------------------
# Summary ranking plot
# -----------------------------------------

def plot_best_scores(df):

    best_rows = []

    for model in df["Model"].unique():

        subset = df[df["Model"] == model]

        idx = subset["AUC"].idxmax()

        best_rows.append(subset.loc[idx])

    best_df = pd.DataFrame(best_rows)

    plt.figure(figsize=(8,5))

    plt.bar(
        best_df["Model"],
        best_df["AUC"]
    )

    plt.ylabel("Best AUC")
    plt.title(
        "Best AUC per Model"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR/"best_auc.png"
    )

    plt.close()


# -----------------------------------------
# Main
# -----------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    data = load_ablation(INPUT_FILE)

    df = flatten_ablation(data)

    df.to_csv(
        OUTPUT_DIR/"ablation_table.csv",
        index=False
    )

    print(df)

    for metric in METRICS:

        plot_metric(
            df,
            metric
        )

    plot_best_scores(df)

    print(
        f"Saved plots to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()