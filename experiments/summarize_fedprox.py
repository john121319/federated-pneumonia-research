import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    FEDPROX_MU,
    FIGURES_DIR,
    LOCAL_EPOCHS,
    TABLES_DIR,
)
from src.metrics import save_json


parser = argparse.ArgumentParser(
    description="Summarize three FedProx seeds."
)

parser.add_argument(
    "--partition",
    type=str,
    required=True,
    choices=[
        "iid",
        "alpha_05",
        "alpha_01",
    ],
)

parser.add_argument(
    "--mu",
    type=float,
    default=FEDPROX_MU,
)

parser.add_argument(
    "--local-epochs",
    type=int,
    default=LOCAL_EPOCHS,
)

arguments = parser.parse_args()

PARTITION_SCHEME = arguments.partition
MU = float(arguments.mu)
RUN_LOCAL_EPOCHS = int(
    arguments.local_epochs
)

if MU <= 0.0:
    raise ValueError(
        "FedProx mu must be greater than zero."
    )

if RUN_LOCAL_EPOCHS <= 0:
    raise ValueError(
        "Local epochs must be positive."
    )

PARTITION_LABELS = {
    "iid": "IID",
    "alpha_05": "moderate non-IID, alpha=0.5",
    "alpha_01": "severe non-IID, alpha=0.1",
}

PARTITION_LABEL = PARTITION_LABELS[
    PARTITION_SCHEME
]

MU_TAG = (
    format(MU, ".12g")
    .replace("-", "m")
    .replace(".", "p")
    .replace("+", "")
)

EXPERIMENT_PREFIX = (
    f"fedprox_{PARTITION_SCHEME}"
    f"_mu_{MU_TAG}"
    f"_le_{RUN_LOCAL_EPOCHS}"
)

SEEDS = [
    11,
    22,
    33,
]

SUMMARY_PATH = (
    TABLES_DIR
    / f"{EXPERIMENT_PREFIX}_validation_summary.csv"
)

AGGREGATE_PATH = (
    TABLES_DIR
    / f"{EXPERIMENT_PREFIX}_validation_aggregate.csv"
)

REPORT_PATH = (
    TABLES_DIR
    / f"{EXPERIMENT_PREFIX}_validation_report.json"
)

CONVERGENCE_FIGURE_PATH = (
    FIGURES_DIR
    / (
        f"{EXPERIMENT_PREFIX}"
        "_three_seed_pr_auc_convergence.png"
    )
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

seed_reports = []
summary_rows = []

for seed in SEEDS:
    experiment_name = (
        f"{EXPERIMENT_PREFIX}_seed_{seed}"
    )

    seed_report_path = (
        TABLES_DIR
        / f"{experiment_name}_report.json"
    )

    if not seed_report_path.exists():
        raise FileNotFoundError(
            seed_report_path
        )

    with seed_report_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        seed_report = json.load(file)

    if seed_report["algorithm"] != "FedProx":
        raise ValueError(
            f"{seed_report_path} is not a FedProx report."
        )

    if (
        seed_report["partition_scheme"]
        != PARTITION_SCHEME
    ):
        raise ValueError(
            f"Partition mismatch in {seed_report_path}."
        )

    if int(seed_report["seed"]) != seed:
        raise ValueError(
            f"Seed mismatch in {seed_report_path}."
        )

    if not np.isclose(
        float(seed_report["fedprox_mu"]),
        MU,
    ):
        raise ValueError(
            f"Mu mismatch in {seed_report_path}."
        )

    if (
        int(seed_report["local_epochs"])
        != RUN_LOCAL_EPOCHS
    ):
        raise ValueError(
            f"Local-epoch mismatch in {seed_report_path}."
        )

    if (
        seed_report["test_set_used"]
        is not False
    ):
        raise RuntimeError(
            f"{seed_report_path} used the test set."
        )

    selected_metrics = seed_report[
        "best_metrics_selected_threshold"
    ]

    update_statistics = seed_report[
        "best_round_update_statistics"
    ]

    summary_rows.append(
        {
            "seed": int(seed),
            "best_round": int(
                seed_report["best_round"]
            ),
            "fedprox_mu": float(MU),
            "local_epochs": int(
                RUN_LOCAL_EPOCHS
            ),
            "selected_threshold": float(
                seed_report[
                    "selected_validation_threshold"
                ]
            ),
            "accuracy": float(
                selected_metrics["accuracy"]
            ),
            "precision": float(
                selected_metrics["precision"]
            ),
            "sensitivity": float(
                selected_metrics["sensitivity"]
            ),
            "specificity": float(
                selected_metrics["specificity"]
            ),
            "f1_score": float(
                selected_metrics["f1_score"]
            ),
            "balanced_accuracy": float(
                selected_metrics[
                    "balanced_accuracy"
                ]
            ),
            "roc_auc": float(
                selected_metrics["roc_auc"]
            ),
            "pr_auc": float(
                selected_metrics["pr_auc"]
            ),
            "average_precision": float(
                selected_metrics[
                    "average_precision"
                ]
            ),
            "log_loss": float(
                selected_metrics["log_loss"]
            ),
            "best_round_global_update_l2": float(
                update_statistics[
                    "global_update_l2"
                ]
            ),
            "best_round_mean_client_update_l2": float(
                update_statistics[
                    "mean_client_update_l2"
                ]
            ),
            "best_round_mean_proximal_loss": float(
                update_statistics[
                    "mean_client_proximal_loss"
                ]
            ),
            "training_minutes": float(
                seed_report[
                    "total_training_minutes"
                ]
            ),
        }
    )

    seed_reports.append(seed_report)

summary_dataframe = pd.DataFrame(
    summary_rows
)

summary_dataframe.to_csv(
    SUMMARY_PATH,
    index=False,
)

aggregate_metrics = [
    "selected_threshold",
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "average_precision",
    "log_loss",
    "best_round_global_update_l2",
    "best_round_mean_client_update_l2",
    "best_round_mean_proximal_loss",
    "training_minutes",
]

aggregate_rows = []

for metric in aggregate_metrics:
    values = summary_dataframe[
        metric
    ].astype(float)

    aggregate_rows.append(
        {
            "metric": metric,
            "mean": float(values.mean()),
            "standard_deviation": float(
                values.std(ddof=1)
            ),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "number_of_seeds": int(
                len(values)
            ),
        }
    )

aggregate_dataframe = pd.DataFrame(
    aggregate_rows
)

aggregate_dataframe.to_csv(
    AGGREGATE_PATH,
    index=False,
)

plt.figure(
    figsize=(9, 5)
)

for seed_report in seed_reports:
    round_history_dataframe = pd.DataFrame(
        seed_report["round_history"]
    )

    plt.plot(
        round_history_dataframe["round"],
        round_history_dataframe["pr_auc"],
        marker="o",
        label=f"Seed {seed_report['seed']}",
    )

plt.xlabel("Federated round")
plt.ylabel("Validation PR-AUC")

plt.title(
    f"FedProx {PARTITION_LABEL}, mu={MU:g} "
    "validation PR-AUC across three seeds"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    CONVERGENCE_FIGURE_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

combined_report = {
    "experiment_group": (
        EXPERIMENT_PREFIX
    ),
    "algorithm": "FedProx",
    "partition_scheme": (
        PARTITION_SCHEME
    ),
    "partition_label": (
        PARTITION_LABEL
    ),
    "fedprox_mu": float(MU),
    "local_epochs": int(
        RUN_LOCAL_EPOCHS
    ),
    "seeds": SEEDS,
    "number_of_clients": int(
        seed_reports[0][
            "number_of_clients"
        ]
    ),
    "federated_rounds": int(
        seed_reports[0][
            "federated_rounds"
        ]
    ),
    "learning_rate": float(
        seed_reports[0][
            "learning_rate"
        ]
    ),
    "proximal_penalty_scope": (
        "trainable_variables_only"
    ),
    "test_set_used": False,
    "seed_reports": seed_reports,
    "summary_rows": summary_rows,
    "aggregate_metrics": aggregate_rows,
}

save_json(
    combined_report,
    REPORT_PATH,
)

print(
    f"\nFEDPROX {PARTITION_SCHEME} "
    "THREE-SEED SUMMARY"
)

print("=" * 44)

print("\nPER-SEED RESULTS")

display_columns = [
    "seed",
    "best_round",
    "fedprox_mu",
    "selected_threshold",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "f1_score",
    "best_round_global_update_l2",
    "best_round_mean_client_update_l2",
    "best_round_mean_proximal_loss",
    "training_minutes",
]

print(
    summary_dataframe[
        display_columns
    ].to_string(
        index=False
    )
)

print("\nAGGREGATE RESULTS")

print(
    aggregate_dataframe.to_string(
        index=False
    )
)

print("\nGENERATED FILES")
print("===============")
print("Summary:", SUMMARY_PATH)
print("Aggregate:", AGGREGATE_PATH)
print("Combined report:", REPORT_PATH)
print(
    "Combined convergence:",
    CONVERGENCE_FIGURE_PATH,
)
print("\nTest set used:", False)
