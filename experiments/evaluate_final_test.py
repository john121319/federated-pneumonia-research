import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    BATCH_SIZE,
    FIGURES_DIR,
    MANIFEST_DIR,
    RAW_RESULTS_DIR,
    TABLES_DIR,
)
from src.data import (
    create_dataset_from_dataframe,
)
from src.metrics import (
    calculate_binary_metrics,
)


parser = argparse.ArgumentParser(
    description=(
        "Run the single frozen final "
        "test-set evaluation."
    )
)

parser.add_argument(
    "--confirm-final-test",
    action="store_true",
)

arguments = parser.parse_args()

if not arguments.confirm_final_test:
    raise RuntimeError(
        "Final test evaluation was not confirmed. "
        "Run with --confirm-final-test only after "
        "the frozen protocol has been reviewed."
    )

PROTOCOL_PATH = (
    TABLES_DIR
    / "final_test_protocol.json"
)

FINAL_REPORT_PATH = (
    TABLES_DIR
    / "final_test_report.json"
)

PER_SEED_PATH = (
    TABLES_DIR
    / "final_test_per_seed_results.csv"
)

AGGREGATE_PATH = (
    TABLES_DIR
    / "final_test_aggregate_results.csv"
)

PREDICTIONS_PATH = (
    RAW_RESULTS_DIR
    / "final_test_predictions.csv"
)

COMPLETION_MARKER_PATH = (
    TABLES_DIR
    / "final_test_evaluation_completed.lock"
)

FIGURE_PATHS = {
    "roc_auc": (
        FIGURES_DIR
        / "final_test_roc_auc_comparison.png"
    ),
    "pr_auc": (
        FIGURES_DIR
        / "final_test_pr_auc_comparison.png"
    ),
    "balanced_accuracy": (
        FIGURES_DIR
        / (
            "final_test_balanced_accuracy"
            "_comparison.png"
        )
    ),
    "f1_score": (
        FIGURES_DIR
        / "final_test_f1_score_comparison.png"
    ),
}

final_paths = [
    FINAL_REPORT_PATH,
    PER_SEED_PATH,
    AGGREGATE_PATH,
    PREDICTIONS_PATH,
    COMPLETION_MARKER_PATH,
    *FIGURE_PATHS.values(),
]

existing_final_paths = [
    path
    for path in final_paths
    if path.exists()
]

if existing_final_paths:
    raise FileExistsError(
        "Final test outputs already exist. "
        "Do not repeat the final evaluation:\n"
        + "\n".join(
            str(path)
            for path in existing_final_paths
        )
    )

if not PROTOCOL_PATH.exists():
    raise FileNotFoundError(
        PROTOCOL_PATH
    )

for directory in [
    TABLES_DIR,
    RAW_RESULTS_DIR,
    FIGURES_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def to_python_value(value):
    if isinstance(
        value,
        (
            np.integer,
            np.floating,
        ),
    ):
        return value.item()

    return value


with PROTOCOL_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    protocol = json.load(file)

if protocol["test_set_used"] is not False:
    raise RuntimeError(
        "Frozen protocol is already marked "
        "as having used the test set."
    )

if protocol[
    "test_threshold_tuning_allowed"
] is not False:
    raise RuntimeError(
        "Frozen protocol allows test threshold tuning."
    )

if protocol[
    "test_model_selection_allowed"
] is not False:
    raise RuntimeError(
        "Frozen protocol allows test model selection."
    )

condition_records = protocol[
    "conditions"
]

if len(condition_records) != 18:
    raise RuntimeError(
        "Frozen protocol must contain 18 models."
    )

for record in condition_records:
    model_path = Path(
        record["model_path"]
    )

    report_path = Path(
        record[
            "validation_report_path"
        ]
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    if not report_path.exists():
        raise FileNotFoundError(
            report_path
        )

    observed_model_hash = (
        calculate_sha256(
            model_path
        )
    )

    observed_report_hash = (
        calculate_sha256(
            report_path
        )
    )

    if (
        observed_model_hash
        != record["model_sha256"]
    ):
        raise RuntimeError(
            f"Frozen model changed: {model_path}"
        )

    if (
        observed_report_hash
        != record[
            "validation_report_sha256"
        ]
    ):
        raise RuntimeError(
            f"Validation report changed: "
            f"{report_path}"
        )

test_manifest_path = Path(
    protocol["test_manifest_path"]
)

if not test_manifest_path.exists():
    raise FileNotFoundError(
        test_manifest_path
    )

train_manifest_path = (
    MANIFEST_DIR
    / "train_cached.csv"
)

validation_manifest_path = (
    MANIFEST_DIR
    / "validation_cached.csv"
)

for path in [
    train_manifest_path,
    validation_manifest_path,
]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

dtype_mapping = {
    "exam_id": str,
    "original_patient_id": str,
    "cache_path": str,
}

train_dataframe = pd.read_csv(
    train_manifest_path,
    dtype=dtype_mapping,
)

validation_dataframe = pd.read_csv(
    validation_manifest_path,
    dtype=dtype_mapping,
)

test_dataframe = pd.read_csv(
    test_manifest_path,
    dtype=dtype_mapping,
)

required_columns = {
    "exam_id",
    "original_patient_id",
    "detailed_class",
    "label",
    "view_position",
    "cache_path",
}

missing_columns = (
    required_columns
    - set(test_dataframe.columns)
)

if missing_columns:
    raise ValueError(
        "Test manifest is missing columns: "
        f"{sorted(missing_columns)}"
    )

if test_dataframe["exam_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate test examination IDs detected."
    )

train_patient_ids = set(
    train_dataframe[
        "original_patient_id"
    ].astype(str)
)

validation_patient_ids = set(
    validation_dataframe[
        "original_patient_id"
    ].astype(str)
)

test_patient_ids = set(
    test_dataframe[
        "original_patient_id"
    ].astype(str)
)

train_test_overlap = len(
    train_patient_ids
    & test_patient_ids
)

validation_test_overlap = len(
    validation_patient_ids
    & test_patient_ids
)

if train_test_overlap != 0:
    raise RuntimeError(
        "Training-test patient overlap detected."
    )

if validation_test_overlap != 0:
    raise RuntimeError(
        "Validation-test patient overlap detected."
    )

test_true_labels = (
    test_dataframe["label"]
    .astype(np.int32)
    .to_numpy()
)

test_dataset = (
    create_dataset_from_dataframe(
        dataframe=test_dataframe,
        training=False,
        seed=0,
        batch_size=BATCH_SIZE,
    )
)

prediction_columns = [
    "exam_id",
    "original_patient_id",
    "detailed_class",
    "label",
    "view_position",
]

predictions_dataframe = (
    test_dataframe[
        prediction_columns
    ]
    .copy()
)

per_seed_rows = []

print(
    "\nSINGLE FROZEN FINAL TEST EVALUATION"
)

print("=" * 49)

print(
    "Test images:",
    len(test_dataframe),
)

print(
    "Test patients:",
    test_dataframe[
        "original_patient_id"
    ].nunique(),
)

print(
    "Training-test patient overlap:",
    train_test_overlap,
)

print(
    "Validation-test patient overlap:",
    validation_test_overlap,
)

print(
    "Frozen models:",
    len(condition_records),
)

for model_index, record in enumerate(
    condition_records,
    start=1,
):
    model_path = Path(
        record["model_path"]
    )

    threshold = float(
        record[
            "selected_validation_threshold"
        ]
    )

    tf.keras.backend.clear_session()
    gc.collect()

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    probabilities = (
        model.predict(
            test_dataset,
            verbose=0,
        )
        .reshape(-1)
    )

    if len(probabilities) != len(
        test_true_labels
    ):
        raise RuntimeError(
            f"Prediction-count mismatch for "
            f"{model_path}."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            f"Non-finite predictions for "
            f"{model_path}."
        )

    metrics = calculate_binary_metrics(
        true_labels=test_true_labels,
        probabilities=probabilities,
        threshold=threshold,
    )

    probability_column = (
        f"{record['condition_id']}"
        f"__seed_{record['seed']}"
        "__probability"
    )

    prediction_column = (
        f"{record['condition_id']}"
        f"__seed_{record['seed']}"
        "__prediction"
    )

    predictions_dataframe[
        probability_column
    ] = probabilities

    predictions_dataframe[
        prediction_column
    ] = (
        probabilities
        >= threshold
    ).astype(np.int32)

    row = {
        "condition_id": (
            record["condition_id"]
        ),
        "condition_label": (
            record["condition_label"]
        ),
        "algorithm": (
            record["algorithm"]
        ),
        "partition": (
            record["partition"]
        ),
        "seed": int(
            record["seed"]
        ),
        "selected_validation_threshold": float(
            threshold
        ),
        "selected_checkpoint_type": (
            record[
                "selected_checkpoint"
            ]["type"]
        ),
        "selected_checkpoint_value": int(
            record[
                "selected_checkpoint"
            ]["value"]
        ),
    }

    for key, value in metrics.items():
        row[key] = to_python_value(
            value
        )

    per_seed_rows.append(
        row
    )

    print(
        f"[{model_index:02d}/"
        f"{len(condition_records):02d}] "
        f"{record['condition_label']}, "
        f"seed={record['seed']}, "
        f"ROC-AUC={metrics['roc_auc']:.6f}, "
        f"PR-AUC={metrics['pr_auc']:.6f}, "
        f"BA="
        f"{metrics['balanced_accuracy']:.6f}, "
        f"F1={metrics['f1_score']:.6f}"
    )

    del model
    gc.collect()

per_seed_dataframe = pd.DataFrame(
    per_seed_rows
)

condition_order = [
    "centralized",
    "fedavg_iid",
    "fedavg_alpha_05",
    "fedavg_alpha_01",
    "fedprox_alpha_05",
    "fedprox_alpha_01",
]

condition_labels = {
    "centralized": "Centralized",
    "fedavg_iid": "FedAvg IID",
    "fedavg_alpha_05": (
        "FedAvg moderate"
    ),
    "fedavg_alpha_01": (
        "FedAvg severe"
    ),
    "fedprox_alpha_05": (
        "FedProx moderate"
    ),
    "fedprox_alpha_01": (
        "FedProx severe"
    ),
}

aggregate_metric_names = [
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
]

aggregate_rows = []

for condition_id in condition_order:
    condition_dataframe = (
        per_seed_dataframe[
            per_seed_dataframe[
                "condition_id"
            ]
            == condition_id
        ]
    )

    if len(condition_dataframe) != 3:
        raise RuntimeError(
            f"{condition_id} does not have "
            "three seed results."
        )

    row = {
        "condition_id": condition_id,
        "condition_label": (
            condition_dataframe[
                "condition_label"
            ].iloc[0]
        ),
        "algorithm": (
            condition_dataframe[
                "algorithm"
            ].iloc[0]
        ),
        "partition": (
            condition_dataframe[
                "partition"
            ].iloc[0]
        ),
        "number_of_seeds": int(
            len(condition_dataframe)
        ),
    }

    for metric_name in aggregate_metric_names:
        values = condition_dataframe[
            metric_name
        ].astype(float)

        row[
            f"{metric_name}_mean"
        ] = float(
            values.mean()
        )

        row[
            f"{metric_name}_standard_deviation"
        ] = float(
            values.std(ddof=1)
        )

        row[
            f"{metric_name}_minimum"
        ] = float(
            values.min()
        )

        row[
            f"{metric_name}_maximum"
        ] = float(
            values.max()
        )

    aggregate_rows.append(
        row
    )

aggregate_dataframe = pd.DataFrame(
    aggregate_rows
)

for metric_name, figure_path in (
    FIGURE_PATHS.items()
):
    means = []
    standard_deviations = []
    labels = []

    for condition_id in condition_order:
        row = aggregate_dataframe[
            aggregate_dataframe[
                "condition_id"
            ]
            == condition_id
        ].iloc[0]

        means.append(
            row[
                f"{metric_name}_mean"
            ]
        )

        standard_deviations.append(
            row[
                f"{metric_name}"
                "_standard_deviation"
            ]
        )

        labels.append(
            condition_labels[
                condition_id
            ]
        )

    x_positions = np.arange(
        len(condition_order)
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        x_positions,
        means,
        yerr=standard_deviations,
        capsize=5,
    )

    plt.xticks(
        x_positions,
        labels,
        rotation=25,
        ha="right",
    )

    plt.ylabel(
        metric_name.replace(
            "_",
            " ",
        ).upper()
        if "auc" in metric_name
        else metric_name.replace(
            "_",
            " ",
        ).title()
    )

    plt.title(
        "Frozen final test comparison: "
        + metric_name.replace(
            "_",
            " ",
        )
    )

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

predictions_dataframe.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

per_seed_dataframe.to_csv(
    PER_SEED_PATH,
    index=False,
)

aggregate_dataframe.to_csv(
    AGGREGATE_PATH,
    index=False,
)

protocol_hash = calculate_sha256(
    PROTOCOL_PATH
)

report = {
    "experiment_type": (
        "single_frozen_final_test_evaluation"
    ),
    "completed_at_utc": (
        datetime.now(
            timezone.utc
        ).isoformat()
    ),
    "protocol_path": str(
        PROTOCOL_PATH
    ),
    "protocol_sha256": (
        protocol_hash
    ),
    "test_manifest_path": str(
        test_manifest_path
    ),
    "test_images": int(
        len(test_dataframe)
    ),
    "test_patients": int(
        test_dataframe[
            "original_patient_id"
        ].nunique()
    ),
    "positive_test_images": int(
        test_dataframe["label"].sum()
    ),
    "negative_test_images": int(
        len(test_dataframe)
        - test_dataframe["label"].sum()
    ),
    "training_test_patient_overlap": int(
        train_test_overlap
    ),
    "validation_test_patient_overlap": int(
        validation_test_overlap
    ),
    "threshold_source": (
        "frozen_validation_selected_threshold"
    ),
    "threshold_tuned_on_test": False,
    "model_selected_on_test": False,
    "number_of_models": int(
        len(condition_records)
    ),
    "number_of_conditions": int(
        len(condition_order)
    ),
    "seeds": protocol["seeds"],
    "per_seed_results": (
        per_seed_rows
    ),
    "aggregate_results": (
        aggregate_rows
    ),
    "generated_files": {
        "per_seed_results": str(
            PER_SEED_PATH
        ),
        "aggregate_results": str(
            AGGREGATE_PATH
        ),
        "predictions": str(
            PREDICTIONS_PATH
        ),
        "figures": {
            key: str(value)
            for key, value
            in FIGURE_PATHS.items()
        },
    },
    "test_set_used": True,
    "final_evaluation_repeated": False,
}

temporary_report_path = (
    FINAL_REPORT_PATH.with_suffix(
        ".json.tmp"
    )
)

with temporary_report_path.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        report,
        file,
        indent=2,
    )

temporary_report_path.replace(
    FINAL_REPORT_PATH
)

with COMPLETION_MARKER_PATH.open(
    "x",
    encoding="utf-8",
) as file:
    file.write(
        protocol_hash
        + "\n"
    )

print(
    "\nFINAL TEST AGGREGATE RESULTS"
)

print("=" * 38)

display_columns = [
    "condition_label",
    "roc_auc_mean",
    "roc_auc_standard_deviation",
    "pr_auc_mean",
    "pr_auc_standard_deviation",
    "balanced_accuracy_mean",
    "balanced_accuracy_standard_deviation",
    "f1_score_mean",
    "f1_score_standard_deviation",
]

print(
    aggregate_dataframe[
        display_columns
    ].to_string(
        index=False
    )
)

print("\nGENERATED FILES")
print("===============")
print("Protocol:", PROTOCOL_PATH)
print("Per-seed results:", PER_SEED_PATH)
print("Aggregate results:", AGGREGATE_PATH)
print("Predictions:", PREDICTIONS_PATH)
print("Final report:", FINAL_REPORT_PATH)

for key, value in FIGURE_PATHS.items():
    print(
        f"{key} figure:",
        value,
    )

print(
    "Completion marker:",
    COMPLETION_MARKER_PATH,
)

print("\nThreshold tuned on test:", False)
print("Model selected on test:", False)
print("Final evaluation repeated:", False)
print("Test set used:", True)
