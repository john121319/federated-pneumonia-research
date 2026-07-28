import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, TABLES_DIR
from src.metrics import save_json

PARTITIONS = [
    "iid",
    "alpha_05",
    "alpha_01",
]

PARTITION_LABELS = {
    "iid": "IID",
    "alpha_05": "Moderate non-IID, alpha=0.5",
    "alpha_01": "Severe non-IID, alpha=0.1",
}

METRICS = [
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "f1_score",
]

SUMMARY_PATH = (
    TABLES_DIR
    / "fedavg_condition_comparison.csv"
)

PAIRED_DIFFERENCE_PATH = (
    TABLES_DIR
    / "fedavg_condition_paired_differences.csv"
)

REPORT_PATH = (
    TABLES_DIR
    / "fedavg_condition_comparison_report.json"
)

FIGURE_PATH = (
    FIGURES_DIR
    / "fedavg_condition_metric_comparison.png"
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

condition_rows = []
seed_dataframes = {}

for partition in PARTITIONS:
    aggregate_path = (
        TABLES_DIR
        / f"fedavg_{partition}_validation_aggregate.csv"
    )

    seed_summary_path = (
        TABLES_DIR
        / f"fedavg_{partition}_validation_summary.csv"
    )

    report_path = (
        TABLES_DIR
        / f"fedavg_{partition}_validation_report.json"
    )

    if not aggregate_path.exists():
        raise FileNotFoundError(
            aggregate_path
        )

    if not seed_summary_path.exists():
        raise FileNotFoundError(
            seed_summary_path
        )

    if not report_path.exists():
        raise FileNotFoundError(
            report_path
        )

    aggregate_dataframe = pd.read_csv(
        aggregate_path
    )

    seed_dataframe = pd.read_csv(
        seed_summary_path
    )

    with report_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(
            file
        )

    if report[
        "algorithm"
    ] != "FedAvg":
        raise ValueError(
            f"{report_path} is not a FedAvg report."
        )

    if report[
        "partition_scheme"
    ] != partition:
        raise ValueError(
            f"Partition mismatch in {report_path}."
        )

    if report[
        "test_set_used"
    ] is not False:
        raise RuntimeError(
            f"{report_path} unexpectedly used the test set."
        )

    required_aggregate_columns = {
        "metric",
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
        "number_of_seeds",
    }

    missing_aggregate_columns = (
        required_aggregate_columns
        - set(
            aggregate_dataframe.columns
        )
    )

    if missing_aggregate_columns:
        raise ValueError(
            f"{aggregate_path} is missing columns: "
            f"{sorted(missing_aggregate_columns)}"
        )

    required_seed_columns = {
        "seed",
        *METRICS,
    }

    missing_seed_columns = (
        required_seed_columns
        - set(
            seed_dataframe.columns
        )
    )

    if missing_seed_columns:
        raise ValueError(
            f"{seed_summary_path} is missing columns: "
            f"{sorted(missing_seed_columns)}"
        )

    if sorted(
        seed_dataframe[
            "seed"
        ].astype(int).tolist()
    ) != [
        11,
        22,
        33,
    ]:
        raise ValueError(
            f"Unexpected seeds in {seed_summary_path}."
        )

    aggregate_indexed = (
        aggregate_dataframe
        .set_index(
            "metric"
        )
    )

    row = {
        "partition_scheme": partition,
        "condition": PARTITION_LABELS[
            partition
        ],
    }

    for metric in METRICS:
        if metric not in aggregate_indexed.index:
            raise ValueError(
                f"{metric} is missing from {aggregate_path}."
            )

        row[
            f"{metric}_mean"
        ] = float(
            aggregate_indexed.loc[
                metric,
                "mean",
            ]
        )

        row[
            f"{metric}_standard_deviation"
        ] = float(
            aggregate_indexed.loc[
                metric,
                "standard_deviation",
            ]
        )

    condition_rows.append(
        row
    )

    seed_dataframes[
        partition
    ] = (
        seed_dataframe[
            [
                "seed",
                *METRICS,
            ]
        ]
        .copy()
        .sort_values(
            "seed"
        )
        .reset_index(
            drop=True
        )
    )

comparison_dataframe = pd.DataFrame(
    condition_rows
)

comparison_dataframe.to_csv(
    SUMMARY_PATH,
    index=False,
)

paired_comparisons = [
    (
        "alpha_05_minus_iid",
        "alpha_05",
        "iid",
    ),
    (
        "alpha_01_minus_iid",
        "alpha_01",
        "iid",
    ),
    (
        "alpha_01_minus_alpha_05",
        "alpha_01",
        "alpha_05",
    ),
]

paired_rows = []

for comparison_name, first_partition, second_partition in paired_comparisons:
    first_dataframe = seed_dataframes[
        first_partition
    ]

    second_dataframe = seed_dataframes[
        second_partition
    ]

    merged_dataframe = first_dataframe.merge(
        second_dataframe,
        on="seed",
        suffixes=(
            "_first",
            "_second",
        ),
        validate="one_to_one",
    )

    for metric in METRICS:
        differences = (
            merged_dataframe[
                f"{metric}_first"
            ]
            - merged_dataframe[
                f"{metric}_second"
            ]
        )

        for seed, difference in zip(
            merged_dataframe[
                "seed"
            ],
            differences,
        ):
            paired_rows.append(
                {
                    "comparison": comparison_name,
                    "first_condition": PARTITION_LABELS[
                        first_partition
                    ],
                    "second_condition": PARTITION_LABELS[
                        second_partition
                    ],
                    "metric": metric,
                    "seed": int(
                        seed
                    ),
                    "difference": float(
                        difference
                    ),
                    "mean_difference": float(
                        differences.mean()
                    ),
                    "standard_deviation": float(
                        differences.std(
                            ddof=1
                        )
                    ),
                }
            )

paired_dataframe = pd.DataFrame(
    paired_rows
)

paired_dataframe.to_csv(
    PAIRED_DIFFERENCE_PATH,
    index=False,
)

x_values = np.arange(
    len(
        PARTITIONS
    )
)

plt.figure(
    figsize=(
        10,
        6,
    )
)

for metric in METRICS:
    means = [
        float(
            comparison_dataframe.loc[
                comparison_dataframe[
                    "partition_scheme"
                ] == partition,
                f"{metric}_mean",
            ].iloc[
                0
            ]
        )
        for partition
        in PARTITIONS
    ]

    standard_deviations = [
        float(
            comparison_dataframe.loc[
                comparison_dataframe[
                    "partition_scheme"
                ] == partition,
                f"{metric}_standard_deviation",
            ].iloc[
                0
            ]
        )
        for partition
        in PARTITIONS
    ]

    plt.errorbar(
        x_values,
        means,
        yerr=standard_deviations,
        marker="o",
        capsize=4,
        linewidth=2,
        label=metric.replace(
            "_",
            " ",
        ).upper(),
    )

plt.xticks(
    x_values,
    [
        "IID",
        "Moderate non-IID\nalpha=0.5",
        "Severe non-IID\nalpha=0.1",
    ],
)

plt.xlabel(
    "Federated client-data condition"
)

plt.ylabel(
    "Validation metric"
)

plt.title(
    "FedAvg validation performance as client heterogeneity increases"
)

plt.ylim(
    0.50,
    0.85,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

report = {
    "analysis_name": "fedavg_condition_comparison",
    "algorithm": "FedAvg",
    "partitions": PARTITIONS,
    "partition_labels": PARTITION_LABELS,
    "metrics": METRICS,
    "seeds": [
        11,
        22,
        33,
    ],
    "test_set_used": False,
    "condition_summary": (
        condition_rows
    ),
    "paired_differences": (
        paired_rows
    ),
}

save_json(
    report,
    REPORT_PATH,
)

print(
    "\nFEDAVG CONDITION COMPARISON"
)

print(
    "==========================="
)

display_columns = [
    "condition",
]

for metric in METRICS:
    display_columns.extend(
        [
            f"{metric}_mean",
            f"{metric}_standard_deviation",
        ]
    )

print(
    "\nCONDITION SUMMARY"
)

print(
    comparison_dataframe[
        display_columns
    ].to_string(
        index=False
    )
)

print(
    "\nMEAN PAIRED DIFFERENCES"
)

paired_summary = (
    paired_dataframe[
        [
            "comparison",
            "metric",
            "mean_difference",
            "standard_deviation",
        ]
    ]
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)

print(
    paired_summary.to_string(
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
    "Comparison table:",
    SUMMARY_PATH,
)

print(
    "Paired differences:",
    PAIRED_DIFFERENCE_PATH,
)

print(
    "Comparison report:",
    REPORT_PATH,
)

print(
    "Comparison figure:",
    FIGURE_PATH,
)

print(
    "\nTest set used:",
    False,
)
