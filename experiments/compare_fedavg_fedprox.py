import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

SEEDS = [11, 22, 33]

PARTITIONS = {
    "alpha_05": "Moderate non-IID, alpha=0.5",
    "alpha_01": "Severe non-IID, alpha=0.1",
}

ALGORITHMS = ["FedAvg", "FedProx"]

PREDICTIVE_METRICS = [
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "f1_score",
]

THRESHOLD_METRICS = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
]

UPDATE_METRICS = [
    "global_update_l2",
    "mean_client_update_l2",
]

METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "balanced_accuracy": "Balanced accuracy",
    "f1_score": "F1-score",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "global_update_l2": "Global update L2",
    "mean_client_update_l2": "Mean client update L2",
}


def report_path(
    algorithm,
    partition,
    seed,
):
    if algorithm == "FedAvg":
        return (
            TABLES_DIR
            / f"fedavg_{partition}_seed_{seed}_report.json"
        )

    return (
        TABLES_DIR
        / (
            f"fedprox_{partition}"
            f"_mu_0p01_le_1_seed_{seed}_report.json"
        )
    )


def load_report(
    algorithm,
    partition,
    seed,
):
    path = report_path(
        algorithm,
        partition,
        seed,
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    if report["algorithm"] != algorithm:
        raise ValueError(
            f"Algorithm mismatch in {path}: "
            f"{report['algorithm']}"
        )

    if report["partition_scheme"] != partition:
        raise ValueError(
            f"Partition mismatch in {path}: "
            f"{report['partition_scheme']}"
        )

    if int(report["seed"]) != seed:
        raise ValueError(
            f"Seed mismatch in {path}: "
            f"{report['seed']}"
        )

    if int(report["federated_rounds"]) != 20:
        raise ValueError(
            f"Unexpected round count in {path}: "
            f"{report['federated_rounds']}"
        )

    if int(report["local_epochs"]) != 1:
        raise ValueError(
            f"Unexpected local epochs in {path}: "
            f"{report['local_epochs']}"
        )

    if report["test_set_used"] is not False:
        raise RuntimeError(
            f"The test set was used in {path}."
        )

    if int(
        report[
            "training_validation_patient_overlap"
        ]
    ) != 0:
        raise RuntimeError(
            "Training-validation patient overlap "
            f"detected in {path}."
        )

    if int(
        report[
            "patient_overlap_between_clients"
        ]
    ) != 0:
        raise RuntimeError(
            "Patient overlap between clients "
            f"detected in {path}."
        )

    if algorithm == "FedProx":
        if not np.isclose(
            float(report["fedprox_mu"]),
            0.01,
        ):
            raise ValueError(
                f"Unexpected FedProx mu in {path}: "
                f"{report['fedprox_mu']}"
            )

        if (
            report["proximal_penalty_scope"]
            != "trainable_variables_only"
        ):
            raise ValueError(
                "Unexpected proximal-penalty scope "
                f"in {path}."
            )

    return report


def best_round_row(
    report,
):
    best_round = int(
        report["best_round"]
    )

    matching_rows = [
        row
        for row in report["round_history"]
        if int(row["round"]) == best_round
    ]

    if len(matching_rows) != 1:
        raise RuntimeError(
            "Expected exactly one round-history "
            f"row for best round {best_round}."
        )

    return matching_rows[0]


def make_metric_figure(
    summary_dataframe,
    metric,
    output_path,
):
    conditions = [
        "Moderate non-IID",
        "Severe non-IID",
    ]

    partition_order = [
        "alpha_05",
        "alpha_01",
    ]

    x_positions = np.arange(
        len(partition_order)
    )

    width = 0.35

    plt.figure(
        figsize=(8, 5)
    )

    for algorithm_index, algorithm in enumerate(
        ALGORITHMS
    ):
        algorithm_rows = (
            summary_dataframe[
                summary_dataframe["algorithm"]
                == algorithm
            ]
            .set_index("partition")
            .loc[partition_order]
        )

        offsets = (
            x_positions
            + (
                algorithm_index
                - 0.5
            )
            * width
        )

        plt.bar(
            offsets,
            algorithm_rows[
                f"{metric}_mean"
            ],
            width=width,
            yerr=algorithm_rows[
                f"{metric}_standard_deviation"
            ],
            capsize=4,
            label=algorithm,
        )

    plt.xticks(
        x_positions,
        conditions,
    )

    plt.ylabel(
        METRIC_LABELS[metric]
    )

    plt.title(
        f"FedAvg versus FedProx: "
        f"{METRIC_LABELS[metric]}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

per_seed_rows = []
loaded_reports = {}

for algorithm in ALGORITHMS:
    for partition, condition_label in (
        PARTITIONS.items()
    ):
        for seed in SEEDS:
            report = load_report(
                algorithm=algorithm,
                partition=partition,
                seed=seed,
            )

            loaded_reports[
                (
                    algorithm,
                    partition,
                    seed,
                )
            ] = report

            selected_metrics = report[
                "best_metrics_selected_threshold"
            ]

            round_row = best_round_row(
                report
            )

            per_seed_rows.append(
                {
                    "algorithm": algorithm,
                    "partition": partition,
                    "condition": condition_label,
                    "seed": int(seed),
                    "best_round": int(
                        report["best_round"]
                    ),
                    "selected_threshold": float(
                        report[
                            "selected_validation_threshold"
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
                    "balanced_accuracy": float(
                        selected_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "f1_score": float(
                        selected_metrics["f1_score"]
                    ),
                    "accuracy": float(
                        selected_metrics["accuracy"]
                    ),
                    "precision": float(
                        selected_metrics["precision"]
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
                    "log_loss": float(
                        selected_metrics["log_loss"]
                    ),
                    "global_update_l2": float(
                        round_row[
                            "global_update_l2"
                        ]
                    ),
                    "mean_client_update_l2": float(
                        round_row[
                            "mean_client_update_l2"
                        ]
                    ),
                    "training_minutes": float(
                        report[
                            "total_training_minutes"
                        ]
                    ),
                }
            )

per_seed_dataframe = pd.DataFrame(
    per_seed_rows
)

per_seed_dataframe["algorithm"] = pd.Categorical(
    per_seed_dataframe["algorithm"],
    categories=ALGORITHMS,
    ordered=True,
)

per_seed_dataframe["partition"] = pd.Categorical(
    per_seed_dataframe["partition"],
    categories=[
        "alpha_05",
        "alpha_01",
    ],
    ordered=True,
)

per_seed_dataframe = (
    per_seed_dataframe
    .sort_values(
        [
            "partition",
            "algorithm",
            "seed",
        ]
    )
    .reset_index(
        drop=True
    )
)

numeric_columns = [
    "selected_threshold",
    "roc_auc",
    "pr_auc",
    "average_precision",
    "balanced_accuracy",
    "f1_score",
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "log_loss",
    "global_update_l2",
    "mean_client_update_l2",
    "training_minutes",
]

if not np.isfinite(
    per_seed_dataframe[
        numeric_columns
    ].to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Non-finite values detected in "
        "the per-seed comparison table."
    )

PER_SEED_PATH = (
    TABLES_DIR
    / "fedavg_fedprox_per_seed_comparison.csv"
)

per_seed_dataframe.to_csv(
    PER_SEED_PATH,
    index=False,
)

aggregate_metrics = (
    PREDICTIVE_METRICS
    + THRESHOLD_METRICS
    + UPDATE_METRICS
    + [
        "selected_threshold",
        "log_loss",
        "training_minutes",
    ]
)

aggregate_rows = []

for algorithm in ALGORITHMS:
    for partition, condition_label in (
        PARTITIONS.items()
    ):
        condition_rows = (
            per_seed_dataframe[
                (
                    per_seed_dataframe[
                        "algorithm"
                    ]
                    == algorithm
                )
                & (
                    per_seed_dataframe[
                        "partition"
                    ]
                    == partition
                )
            ]
        )

        aggregate_row = {
            "algorithm": algorithm,
            "partition": partition,
            "condition": condition_label,
            "number_of_seeds": int(
                len(condition_rows)
            ),
        }

        for metric in aggregate_metrics:
            values = condition_rows[
                metric
            ].astype(float)

            aggregate_row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            aggregate_row[
                f"{metric}_standard_deviation"
            ] = float(
                values.std(
                    ddof=1
                )
            )

            aggregate_row[
                f"{metric}_minimum"
            ] = float(
                values.min()
            )

            aggregate_row[
                f"{metric}_maximum"
            ] = float(
                values.max()
            )

        aggregate_rows.append(
            aggregate_row
        )

aggregate_dataframe = pd.DataFrame(
    aggregate_rows
)

AGGREGATE_PATH = (
    TABLES_DIR
    / "fedavg_fedprox_aggregate_comparison.csv"
)

aggregate_dataframe.to_csv(
    AGGREGATE_PATH,
    index=False,
)

paired_rows = []

for partition, condition_label in (
    PARTITIONS.items()
):
    for seed in SEEDS:
        fedavg_row = (
            per_seed_dataframe[
                (
                    per_seed_dataframe[
                        "algorithm"
                    ]
                    == "FedAvg"
                )
                & (
                    per_seed_dataframe[
                        "partition"
                    ]
                    == partition
                )
                & (
                    per_seed_dataframe[
                        "seed"
                    ]
                    == seed
                )
            ]
            .iloc[0]
        )

        fedprox_row = (
            per_seed_dataframe[
                (
                    per_seed_dataframe[
                        "algorithm"
                    ]
                    == "FedProx"
                )
                & (
                    per_seed_dataframe[
                        "partition"
                    ]
                    == partition
                )
                & (
                    per_seed_dataframe[
                        "seed"
                    ]
                    == seed
                )
            ]
            .iloc[0]
        )

        for metric in (
            PREDICTIVE_METRICS
            + UPDATE_METRICS
        ):
            fedavg_value = float(
                fedavg_row[metric]
            )

            fedprox_value = float(
                fedprox_row[metric]
            )

            paired_row = {
                "partition": partition,
                "condition": condition_label,
                "seed": int(seed),
                "metric": metric,
                "fedavg_value": fedavg_value,
                "fedprox_value": fedprox_value,
                "fedprox_minus_fedavg": float(
                    fedprox_value
                    - fedavg_value
                ),
            }

            if metric in UPDATE_METRICS:
                paired_row[
                    "fedprox_percent_reduction"
                ] = float(
                    (
                        fedavg_value
                        - fedprox_value
                    )
                    / fedavg_value
                    * 100.0
                )
            else:
                paired_row[
                    "fedprox_percent_reduction"
                ] = np.nan

            paired_rows.append(
                paired_row
            )

paired_dataframe = pd.DataFrame(
    paired_rows
)

PAIRED_PATH = (
    TABLES_DIR
    / "fedavg_fedprox_paired_differences.csv"
)

paired_dataframe.to_csv(
    PAIRED_PATH,
    index=False,
)

paired_summary_rows = []

for partition, condition_label in (
    PARTITIONS.items()
):
    for metric in (
        PREDICTIVE_METRICS
        + UPDATE_METRICS
    ):
        metric_rows = (
            paired_dataframe[
                (
                    paired_dataframe[
                        "partition"
                    ]
                    == partition
                )
                & (
                    paired_dataframe[
                        "metric"
                    ]
                    == metric
                )
            ]
        )

        differences = metric_rows[
            "fedprox_minus_fedavg"
        ].astype(float)

        paired_summary_row = {
            "partition": partition,
            "condition": condition_label,
            "metric": metric,
            "mean_fedprox_minus_fedavg": float(
                differences.mean()
            ),
            "standard_deviation": float(
                differences.std(
                    ddof=1
                )
            ),
            "minimum": float(
                differences.min()
            ),
            "maximum": float(
                differences.max()
            ),
            "number_of_seeds": int(
                len(differences)
            ),
        }

        if metric in UPDATE_METRICS:
            reductions = metric_rows[
                "fedprox_percent_reduction"
            ].astype(float)

            paired_summary_row[
                "mean_fedprox_percent_reduction"
            ] = float(
                reductions.mean()
            )

            paired_summary_row[
                "standard_deviation_percent_reduction"
            ] = float(
                reductions.std(
                    ddof=1
                )
            )
        else:
            paired_summary_row[
                "mean_fedprox_percent_reduction"
            ] = np.nan

            paired_summary_row[
                "standard_deviation_percent_reduction"
            ] = np.nan

        paired_summary_rows.append(
            paired_summary_row
        )

paired_summary_dataframe = pd.DataFrame(
    paired_summary_rows
)

PAIRED_SUMMARY_PATH = (
    TABLES_DIR
    / "fedavg_fedprox_paired_summary.csv"
)

paired_summary_dataframe.to_csv(
    PAIRED_SUMMARY_PATH,
    index=False,
)

heterogeneity_rows = []

for algorithm in ALGORITHMS:
    for seed in SEEDS:
        moderate_row = (
            per_seed_dataframe[
                (
                    per_seed_dataframe[
                        "algorithm"
                    ]
                    == algorithm
                )
                & (
                    per_seed_dataframe[
                        "partition"
                    ]
                    == "alpha_05"
                )
                & (
                    per_seed_dataframe[
                        "seed"
                    ]
                    == seed
                )
            ]
            .iloc[0]
        )

        severe_row = (
            per_seed_dataframe[
                (
                    per_seed_dataframe[
                        "algorithm"
                    ]
                    == algorithm
                )
                & (
                    per_seed_dataframe[
                        "partition"
                    ]
                    == "alpha_01"
                )
                & (
                    per_seed_dataframe[
                        "seed"
                    ]
                    == seed
                )
            ]
            .iloc[0]
        )

        for metric in (
            PREDICTIVE_METRICS
            + UPDATE_METRICS
        ):
            heterogeneity_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": int(seed),
                    "metric": metric,
                    "moderate_value": float(
                        moderate_row[metric]
                    ),
                    "severe_value": float(
                        severe_row[metric]
                    ),
                    "severe_minus_moderate": float(
                        severe_row[metric]
                        - moderate_row[metric]
                    ),
                }
            )

heterogeneity_dataframe = pd.DataFrame(
    heterogeneity_rows
)

HETEROGENEITY_PATH = (
    TABLES_DIR
    / (
        "fedavg_fedprox_heterogeneity"
        "_differences.csv"
    )
)

heterogeneity_dataframe.to_csv(
    HETEROGENEITY_PATH,
    index=False,
)

heterogeneity_summary_rows = []

for algorithm in ALGORITHMS:
    for metric in (
        PREDICTIVE_METRICS
        + UPDATE_METRICS
    ):
        values = (
            heterogeneity_dataframe[
                (
                    heterogeneity_dataframe[
                        "algorithm"
                    ]
                    == algorithm
                )
                & (
                    heterogeneity_dataframe[
                        "metric"
                    ]
                    == metric
                )
            ][
                "severe_minus_moderate"
            ]
            .astype(float)
        )

        heterogeneity_summary_rows.append(
            {
                "algorithm": algorithm,
                "metric": metric,
                "mean_severe_minus_moderate": float(
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
                    len(values)
                ),
            }
        )

heterogeneity_summary_dataframe = pd.DataFrame(
    heterogeneity_summary_rows
)

HETEROGENEITY_SUMMARY_PATH = (
    TABLES_DIR
    / (
        "fedavg_fedprox_heterogeneity"
        "_summary.csv"
    )
)

heterogeneity_summary_dataframe.to_csv(
    HETEROGENEITY_SUMMARY_PATH,
    index=False,
)

figure_paths = {}

for metric in (
    PREDICTIVE_METRICS
    + UPDATE_METRICS
):
    figure_path = (
        FIGURES_DIR
        / (
            "fedavg_fedprox_"
            f"{metric}_comparison.png"
        )
    )

    make_metric_figure(
        summary_dataframe=aggregate_dataframe,
        metric=metric,
        output_path=figure_path,
    )

    figure_paths[
        metric
    ] = str(
        figure_path
    )

report = {
    "experiment_type": (
        "fedavg_fedprox_validation_comparison"
    ),
    "algorithms": ALGORITHMS,
    "partitions": PARTITIONS,
    "seeds": SEEDS,
    "number_of_clients": 5,
    "federated_rounds": 20,
    "local_epochs": 1,
    "fedprox_mu": 0.01,
    "model_selection_metric": (
        "validation_pr_auc"
    ),
    "test_set_used": False,
    "descriptive_analysis_only": True,
    "statistical_significance_claimed": False,
    "per_seed_results": (
        per_seed_dataframe
        .assign(
            algorithm=lambda frame: (
                frame[
                    "algorithm"
                ].astype(str)
            ),
            partition=lambda frame: (
                frame[
                    "partition"
                ].astype(str)
            ),
        )
        .to_dict(
            orient="records"
        )
    ),
    "aggregate_results": (
        aggregate_dataframe.to_dict(
            orient="records"
        )
    ),
    "paired_fedprox_minus_fedavg": (
        paired_dataframe
        .replace(
            {
                np.nan: None
            }
        )
        .to_dict(
            orient="records"
        )
    ),
    "paired_summary": (
        paired_summary_dataframe
        .replace(
            {
                np.nan: None
            }
        )
        .to_dict(
            orient="records"
        )
    ),
    "heterogeneity_differences": (
        heterogeneity_dataframe.to_dict(
            orient="records"
        )
    ),
    "heterogeneity_summary": (
        heterogeneity_summary_dataframe.to_dict(
            orient="records"
        )
    ),
    "generated_files": {
        "per_seed_table": str(
            PER_SEED_PATH
        ),
        "aggregate_table": str(
            AGGREGATE_PATH
        ),
        "paired_differences": str(
            PAIRED_PATH
        ),
        "paired_summary": str(
            PAIRED_SUMMARY_PATH
        ),
        "heterogeneity_differences": str(
            HETEROGENEITY_PATH
        ),
        "heterogeneity_summary": str(
            HETEROGENEITY_SUMMARY_PATH
        ),
        "figures": figure_paths,
    },
}

REPORT_PATH = (
    TABLES_DIR
    / "fedavg_fedprox_comparison_report.json"
)

with REPORT_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        report,
        file,
        indent=2,
        allow_nan=False,
    )

print(
    "\nFEDAVG VERSUS FEDPROX "
    "VALIDATION COMPARISON"
)

print("=" * 48)

display_columns = [
    "algorithm",
    "partition",
    "roc_auc_mean",
    "roc_auc_standard_deviation",
    "pr_auc_mean",
    "pr_auc_standard_deviation",
    "balanced_accuracy_mean",
    "balanced_accuracy_standard_deviation",
    "f1_score_mean",
    "f1_score_standard_deviation",
    "global_update_l2_mean",
    "mean_client_update_l2_mean",
]

print("\nAGGREGATE RESULTS")

print(
    aggregate_dataframe[
        display_columns
    ].to_string(
        index=False
    )
)

print(
    "\nFEDPROX MINUS FEDAVG "
    "PAIRED SUMMARY"
)

print(
    paired_summary_dataframe[
        [
            "partition",
            "metric",
            "mean_fedprox_minus_fedavg",
            "standard_deviation",
            "mean_fedprox_percent_reduction",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nSEVERE MINUS MODERATE "
    "HETEROGENEITY SUMMARY"
)

print(
    heterogeneity_summary_dataframe.to_string(
        index=False
    )
)

print("\nGENERATED FILES")
print("===============")
print("Per-seed table:", PER_SEED_PATH)
print("Aggregate table:", AGGREGATE_PATH)
print("Paired differences:", PAIRED_PATH)
print("Paired summary:", PAIRED_SUMMARY_PATH)
print(
    "Heterogeneity differences:",
    HETEROGENEITY_PATH,
)
print(
    "Heterogeneity summary:",
    HETEROGENEITY_SUMMARY_PATH,
)
print("Combined report:", REPORT_PATH)

for metric, path in figure_paths.items():
    print(
        f"{METRIC_LABELS[metric]} figure:",
        path,
    )

print("\nTest set used:", False)
print(
    "Statistical significance claimed:",
    False,
)
