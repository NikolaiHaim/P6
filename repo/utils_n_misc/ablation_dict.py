import json
from pathlib import Path

ABLATION_FILE = "ablation.json"

METRICS = [
    "AUC",
    "AP",
    "MRR",
    "Hit@50",
    "Recall@50",
    "NDCG@50",
    "Mean rank",
    "Median rank"
]

MODELS = ["n2v", "svd", "gcn", "specemb"]


# ------------------------------------------
# Base helpers
# ------------------------------------------

def initialize_ablation():
    """Create ablation file if it doesn't exist."""
    path = Path(ABLATION_FILE)

    if not path.exists():
        data = {model: {} for model in MODELS}

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Created {ABLATION_FILE}")


def load_ablation():
    with open(ABLATION_FILE) as f:
        return json.load(f)


def save_ablation(data):
    with open(ABLATION_FILE, "w") as f:
        json.dump(data, f, indent=4)


def empty_metric_dict():
    return {metric: None for metric in METRICS}


# ------------------------------------------
# Extraction functions
# ------------------------------------------

def extract_gcn_n2v(results):
    """Extract metrics from GCN/N2V format."""

    return {
        "AUC": results["AUC"],
        "AP": results["Average_Precision"],
        "MRR": results["MRR"],
        "Hit@50": results["Hit@50"],
        "Recall@50": results["Recall@50"],
        "NDCG@50": results["NDCG@50"],
        "Mean rank": results["Mean_Rank"],
        "Median rank": results["Median_Rank"],
        "Training time": results["Inference_Time_Sec"]
    }


def extract_specemb(results):
    """Extract metrics from SpecEmb format (test only)."""

    test = results["test"]

    return {
        "AUC": test["auc"],
        "AP": test["ap"],
        "MRR": test["mrr"],
        "Hit@50": test["hit@50"],
        "Recall@50": test["recall@50"],
        "NDCG@50": test["ndcg@50"],
        "Mean rank": test["mean_rank"],
        "Median rank": test["median_rank"],
        "Training time": test["inference_time_sec"]
    }


# Register extractors
EXTRACTORS = {
    "gcn": extract_gcn_n2v,
    "n2v": extract_gcn_n2v,
    "specemb": extract_specemb,
    "svd": extract_gcn_n2v
}


# ------------------------------------------
# Main population logic
# ------------------------------------------

def add_result(model, hyperparam, metrics):
    data = load_ablation()

    data[model][str(hyperparam)] = {
        **empty_metric_dict(),
        **metrics
    }

    save_ablation(data)


def process_results(results_root):
    """
    Expected structure:

    results/
        n2v/
            dim64.json
            dim128.json

        gcn/
            layers2.json

        specemb/
            emb64.json
    """

    results_root = Path(results_root)

    for model_dir in results_root.iterdir():

        if not model_dir.is_dir():
            continue

        model = model_dir.name.lower()

        if model not in EXTRACTORS:
            print(f"Skipping {model}")
            continue

        extractor = EXTRACTORS[model]

        for result_file in model_dir.glob("*.json"):

            with open(result_file) as f:
                results = json.load(f)

            # Uses filename as hyperparameter key
            hyperparam = result_file.stem

            metrics = extractor(results)

            add_result(model, hyperparam, metrics)

            print(
                f"Added {model}/{hyperparam}"
            )


# ------------------------------------------
# Run
# ------------------------------------------

initialize_ablation()
process_results("ablation")