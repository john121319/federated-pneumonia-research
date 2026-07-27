import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, TABLES_DIR
from src.metrics import save_json

parser = argparse.ArgumentParser(
    description="Summarize three-seed FedAvg validation results."
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

arguments = parser.parse_args()

PARTITION_SCHEME = arguments.partition

PARTITION_LABELS = {
    "iid": "IID",
    "alpha_05": "moderate non-IID, alpha=0.5",
    "alpha_01": "severe non-IID, alpha=0.1",
}

PARTITION_LABEL = PARTITION_LABELS[
    PARTITION_SCHEME
]

SEEDS = [
    11,
    22,
    33,
]

SUMMARY_PATH = (
    TABLES_DIR
    / f"fedavg_{PARTITION_SCHEME}_validation_summary.csv"
)

AGGREGATE_PATH = (
    TABLES_DIR
    / f"fedavg_{PARTITION_SCHEME}_validation_aggregate.csv"
)

REPORT_PATH = (
    TABLES_DIR
    / f"fedavg_{PARTITION_SCHEME}_validation_report.json"
)

CONVERGENCE_PATH = (
    FIGURES_DIR
    / f"fedavg_{PARTITION_SCHEME}_three_seed_pr_auc_convergence.png"
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

summary_rows = []
seed_reports = []
round_dataframes = {}

for seed in SEEDS:
    experiment_name = (
        f"fedavg_{PARTITION_SCHEME}_seed_{seed}"
    )

    seed_report_path = (
        TABLES_DIR
        / f"{experiment_name}_report.json"
    )

    round_history_path = (
        TABLES_DIR
        / f"{experiment_name}_round_history.csv"
    )

    if not seed_report_path.exists():
        raise FileNotFoundError(
            seed_report_path
        )

    if not round_history_path.exists():
        raise FileNotFoundError(
            round_history_path
        )

    with seed_report_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(
            file
        )

    round_dataframe = pd.read_csv(
        round_history_path
    )

    required_round_columns = {
        "round",
        "roc_auc",
        "pr_auc",
        "global_update_l2",
        "mean_client_update_l2",
    }

    missing_columns = (
        required_round_columns
        - set(
            round_dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"Seed {seed} round history is missing "
            f"columns: {sorted(missing_columns)}"
        )

    if int(
        report["seed"]
    ) != seed:
        raise ValueError(
            f"Seed mismatch in {seed_report_path}"
        )

    if report[
        "algorithm"
    ] != "FedAvg":
        raise ValueError(
            f"Seed {seed} is not a FedAvg report."
        )

    if report[
        "partition_scheme"
    ] != PARTITION_SCHEME:
        raise ValueError(
            f"Seed {seed} partition mismatch."
        )

    if report[
        "test_set_used"
    ] is not False:
        raise RuntimeError(
            f"Seed {seed} unexpectedly used the test set."
        )

    if int(
        report[
            "training_validation_patient_overlap"
        ]
    ) != 0:
        raise RuntimeError(
            f"Seed {seed} has training-validation overlap."
        )

    if int(
        report[
            "patient_overlap_between_clients"
        ]
    ) != 0:
        raise RuntimeError(
            f"Seed {seed} has patient overlap between clients."
        )

    selected_metrics = report[
        "best_metrics_selected_threshold"
    ]

    metrics_at_half = report[
        "best_metrics_threshold_0_5"
    ]

    summary_rows.append(
        {
            "seed": int(
                seed
            ),
            "best_round": int(
                report[
                    "best_round"
                ]
            ),
            "federated_rounds": int(
                report[
                    "federated_rounds"
                ]
            ),
            "local_epochs": int(
                report[
                    "local_epochs"
                ]
            ),
            "learning_rate": float(
                report[
                    "learning_rate"
                ]
            ),
            "training_minutes": float(
                report[
                    "total_training_minutes"
                ]
            ),
            "selected_threshold": float(
                report[
                    "selected_validation_threshold"
                ]
            ),
            "accuracy": float(
                selected_metrics[
                    "accuracy"
                ]
            ),
            "precision": float(
                selected_metrics[
                    "precision"
                ]
            ),
            "sensitivity": float(
                selected_metrics[
                    "sensitivity"
                ]
            ),
            "specificity": float(
                selected_metrics[
                    "specificity"
                ]
            ),
            "f1_score": float(
                selected_metrics[
                    "f1_score"
                ]
            ),
            "balanced_accuracy": float(
                selected_metrics[
                    "balanced_accuracy"
                ]
            ),
            "roc_auc": float(
                selected_metrics[
                    "roc_auc"
                ]
            ),
            "pr_auc": float(
                selected_metrics[
                    "pr_auc"
                ]
            ),
            "average_precision": float(
                selected_metrics[
                    "average_precision"
                ]
            ),
            "log_loss": float(
                selected_metrics[
                    "log_loss"
                ]
            ),
            "true_negative": int(
                selected_metrics[
                    "true_negative"
                ]
            ),
            "false_positive": int(
                selected_metrics[
                    "false_positive"
                ]
            ),
            "false_negative": int(
                selected_metrics[
                    "false_negative"
                ]
            ),
            "true_positive": int(
                selected_metrics[
                    "true_positive"
                ]
            ),
            "accuracy_at_0_5": float(
                metrics_at_half[
                    "accuracy"
                ]
            ),
            "sensitivity_at_0_5": float(
                metrics_at_half[
                    "sensitivity"
                ]
            ),
            "specificity_at_0_5": float(
                metrics_at_half[
                    "specificity"
                ]
            ),
            "f1_score_at_0_5": float(
                metrics_at_half[
                    "f1_score"
                ]
            ),
        }
    )

    seed_reports.append(
        report
    )

    round_dataframes[
        seed
    ] = round_dataframe

summary_dataframe = pd.DataFrame(
    summary_rows
)

summary_dataframe = (
    summary_dataframe
    .sort_values(
        "seed"
    )
    .reset_index(
        drop=True
    )
)

summary_dataframe.to_csv(
    SUMMARY_PATH,
    index=False,
)

aggregate_metric_names = [
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
    "training_minutes",
]

aggregate_rows = []

for metric_name in aggregate_metric_names:
    values = (
        summary_dataframe[
            metric_name
        ]
        .astype(float)
    )

    aggregate_rows.append(
        {
            "metric": metric_name,
            "mean": float(
                values.mean()
            ),
            "standard_deviation": float(
                values.std(
                    ddof=1
                )
            ),
            "minimum": float(
                values.min()
            ),
            "maximum": float(
                values.max()
            ),
            "number_of_seeds": int(
                len(
                    values
                )
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
    figsize=(
        9,
        5,
    )
)

for seed in SEEDS:
    round_dataframe = round_dataframes[
        seed
    ]

    plt.plot(
        round_dataframe[
            "round"
        ],
        round_dataframe[
            "pr_auc"
        ],
        marker="o",
        label=f"Seed {seed}",
    )

plt.xlabel(
    "Federated round"
)

plt.ylabel(
    "Validation PR-AUC"
)

plt.title(
    f"FedAvg {PARTITION_LABEL} validation PR-AUC across three seeds"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    CONVERGENCE_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

combined_report = {
    "experiment_group": (
        f"fedavg_{PARTITION_SCHEME}"
    ),
    "algorithm": "FedAvg",
    "partition_scheme": (
        PARTITION_SCHEME
    ),
    "partition_label": (
        PARTITION_LABEL
    ),
    "seeds": [
        int(
            seed
        )
        for seed
        in SEEDS
    ],
    "number_of_clients": 5,
    "federated_rounds": 20,
    "local_epochs": 1,
    "learning_rate": 0.0005,
    "test_set_used": False,
    "seed_reports": (
        seed_reports
    ),
    "aggregate_metrics": (
        aggregate_rows
    ),
}

save_json(
    combined_report,
    REPORT_PATH,
)

print(
    f"\nFEDAVG {PARTITION_SCHEME} THREE-SEED SUMMARY"
)

print(
    "=" * 42
)

print(
    "\nPER-SEED RESULTS"
)

print(
    summary_dataframe[
        [
            "seed",
            "best_round",
            "selected_threshold",
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "f1_score",
            "training_minutes",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nAGGREGATE RESULTS"
)

print(
    aggregate_dataframe.to_string(
        index=False
    )
)

print(
    "\nGENERATED FILES"
)

print(
    "==============="
)

print(
    "Summary:",
    SUMMARY_PATH,
)

print(
    "Aggregate:",
    AGGREGATE_PATH,
)

print(
    "Combined report:",
    REPORT_PATH,
)

print(
    "Combined convergence:",
    CONVERGENCE_PATH,
)

print(
    "\nTest set used:",
    False,
)
