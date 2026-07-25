import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    FIGURES_DIR,
    TABLES_DIR,
)

from src.metrics import save_json

SEEDS = [
    11,
    22,
    33,
]


SUMMARY_PATH = (
    TABLES_DIR
    / "fedavg_iid_validation_summary.csv"
)


AGGREGATE_PATH = (
    TABLES_DIR
    / "fedavg_iid_validation_aggregate.csv"
)


COMBINED_REPORT_PATH = (
    TABLES_DIR
    / "fedavg_iid_validation_report.json"
)


COMBINED_CONVERGENCE_PATH = (
    FIGURES_DIR
    / "fedavg_iid_three_seed_pr_auc_convergence.png"
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
for seed in SEEDS:

    experiment_name = (
        f"fedavg_iid_seed_{seed}"
    )


    report_path = (
        TABLES_DIR
        / f"{experiment_name}_report.json"
    )


    round_history_path = (
        TABLES_DIR
        / f"{experiment_name}_round_history.csv"
    )


    if not report_path.exists():

        raise FileNotFoundError(
            report_path
        )


    if not round_history_path.exists():

        raise FileNotFoundError(
            round_history_path
        )


    with report_path.open(
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


    missing_round_columns = (
        required_round_columns
        - set(
            round_dataframe.columns
        )
    )


    if missing_round_columns:

        raise ValueError(
            f"Seed {seed} round history is missing: "
            f"{sorted(missing_round_columns)}"
        )


    if int(
        report["seed"]
    ) != seed:

        raise ValueError(
            f"Report seed mismatch for seed {seed}."
        )


    if report[
        "partition_scheme"
    ] != "iid":

        raise ValueError(
            f"Seed {seed} is not an IID report."
        )


    if report[
        "algorithm"
    ] != "FedAvg":

        raise ValueError(
            f"Seed {seed} is not a FedAvg report."
        )


    if report[
        "test_set_used"
    ] is not False:

        raise RuntimeError(
            f"Seed {seed} unexpectedly used the test set."
        )


    best_round = int(
        report[
            "best_round"
        ]
    )


    selected_threshold = float(
        report[
            "selected_validation_threshold"
        ]
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
                best_round
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
                selected_threshold
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
    corrected_convergence_path = (
        FIGURES_DIR
        / (
            f"fedavg_iid_seed_{seed}"
            "_convergence.png"
        )
    )


    plt.figure(
        figsize=(
            9,
            5,
        )
    )


    plt.plot(
        round_dataframe[
            "round"
        ],

        round_dataframe[
            "roc_auc"
        ],

        marker="o",

        label="Validation ROC-AUC",
    )


    plt.plot(
        round_dataframe[
            "round"
        ],

        round_dataframe[
            "pr_auc"
        ],

        marker="o",

        label="Validation PR-AUC",
    )


    plt.axvline(
        best_round,
        linestyle="--",
        label=f"Best round {best_round}",
    )


    plt.xlabel(
        "Federated round"
    )


    plt.ylabel(
        "AUC"
    )


    plt.title(
        f"FedAvg IID validation convergence — seed {seed}"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        corrected_convergence_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close()
    corrected_update_path = (
        FIGURES_DIR
        / (
            f"fedavg_iid_seed_{seed}"
            "_global_update.png"
        )
    )


    trained_rounds = round_dataframe[
        round_dataframe[
            "round"
        ] > 0
    ]


    plt.figure(
        figsize=(
            9,
            5,
        )
    )


    plt.plot(
        trained_rounds[
            "round"
        ],

        trained_rounds[
            "global_update_l2"
        ],

        marker="o",

        label="Global update L2",
    )


    plt.plot(
        trained_rounds[
            "round"
        ],

        trained_rounds[
            "mean_client_update_l2"
        ],

        marker="o",

        label="Mean client update L2",
    )


    plt.xlabel(
        "Federated round"
    )


    plt.ylabel(
        "L2 distance"
    )


    plt.title(
        f"FedAvg IID model updates — seed {seed}"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        corrected_update_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close()
    confusion_values = np.array(
        [
            [
                selected_metrics[
                    "true_negative"
                ],

                selected_metrics[
                    "false_positive"
                ],
            ],

            [
                selected_metrics[
                    "false_negative"
                ],

                selected_metrics[
                    "true_positive"
                ],
            ],
        ]
    )


    corrected_confusion_path = (
        FIGURES_DIR
        / (
            f"fedavg_iid_seed_{seed}"
            "_confusion_matrix.png"
        )
    )


    plt.figure(
        figsize=(
            6,
            5,
        )
    )


    plt.imshow(
        confusion_values
    )


    plt.xticks(
        [
            0,
            1,
        ],

        [
            "Predicted negative",
            "Predicted positive",
        ],
    )


    plt.yticks(
        [
            0,
            1,
        ],

        [
            "Actual negative",
            "Actual positive",
        ],
    )


    for row_index in range(
        2
    ):

        for column_index in range(
            2
        ):

            plt.text(
                column_index,
                row_index,
                str(
                    confusion_values[
                        row_index,
                        column_index
                    ]
                ),
                ha="center",
                va="center",
            )


    plt.title(
        "FedAvg IID validation confusion matrix\n"
        f"Seed {seed}, round {best_round}, "
        f"threshold={selected_threshold:.3f}"
    )


    plt.tight_layout()


    plt.savefig(
        corrected_confusion_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close()
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

    round_history_path = (
        TABLES_DIR
        / (
            f"fedavg_iid_seed_{seed}"
            "_round_history.csv"
        )
    )


    round_dataframe = pd.read_csv(
        round_history_path
    )


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
    "FedAvg IID validation PR-AUC across three seeds"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    COMBINED_CONVERGENCE_PATH,
    dpi=200,
    bbox_inches="tight",
)


plt.close()
combined_report = {
    "experiment_group": (
        "fedavg_iid"
    ),

    "algorithm": "FedAvg",

    "partition_scheme": "iid",

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
    COMBINED_REPORT_PATH,
)

print(
    "\nFEDAVG IID THREE-SEED SUMMARY"
)


print(
    "============================="
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
    COMBINED_REPORT_PATH,
)


print(
    "Combined convergence:",
    COMBINED_CONVERGENCE_PATH,
)


print(
    "\nCorrected individual figures generated:",
    True,
)


print(
    "Test set used:",
    False,
)