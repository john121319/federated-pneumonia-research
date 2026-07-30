import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import (
    MANIFEST_DIR,
    MODELS_DIR,
    TABLES_DIR,
)


SEEDS = [
    11,
    22,
    33,
]

PROTOCOL_PATH = (
    TABLES_DIR
    / "final_test_protocol.json"
)

FINAL_REPORT_PATH = (
    TABLES_DIR
    / "final_test_report.json"
)

TEST_MANIFEST_PATH = (
    MANIFEST_DIR
    / "test_cached.csv"
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


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def require_false(value, label):
    if value is not False:
        raise RuntimeError(
            f"{label} must be False."
        )


def require_finite_threshold(
    threshold,
    label,
):
    threshold = float(threshold)

    if not np.isfinite(threshold):
        raise ValueError(
            f"{label} threshold is not finite."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"{label} threshold is outside [0, 1]."
        )

    return threshold


if PROTOCOL_PATH.exists():
    raise FileExistsError(
        f"Frozen protocol already exists: "
        f"{PROTOCOL_PATH}"
    )

if FINAL_REPORT_PATH.exists():
    raise FileExistsError(
        f"Final test report already exists: "
        f"{FINAL_REPORT_PATH}"
    )

if not TEST_MANIFEST_PATH.exists():
    raise FileNotFoundError(
        TEST_MANIFEST_PATH
    )

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

records = []

centralized_report_path = (
    TABLES_DIR
    / "centralized_validation_report.json"
)

if not centralized_report_path.exists():
    raise FileNotFoundError(
        centralized_report_path
    )

centralized_report = load_json(
    centralized_report_path
)

centralized_seed_reports = {
    int(seed_report["seed"]): seed_report
    for seed_report
    in centralized_report["seed_reports"]
}

for seed in SEEDS:
    if seed not in centralized_seed_reports:
        raise KeyError(
            f"Centralized report is missing seed {seed}."
        )

    seed_report = centralized_seed_reports[
        seed
    ]

    experiment_name = str(
        seed_report["experiment_name"]
    )

    expected_experiment_name = (
        f"centralized_main_seed_{seed}"
    )

    if experiment_name != expected_experiment_name:
        raise ValueError(
            f"Unexpected centralized experiment name "
            f"for seed {seed}: {experiment_name}"
        )

    require_false(
        seed_report.get(
            "test_set_used",
            False,
        ),
        f"Centralized seed {seed} test_set_used",
    )

    model_path = (
        MODELS_DIR
        / f"{experiment_name}_best.keras"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    threshold = require_finite_threshold(
        seed_report[
            "selected_validation_threshold"
        ],
        f"Centralized seed {seed}",
    )

    records.append(
        {
            "condition_id": "centralized",
            "condition_label": "Centralized",
            "algorithm": "Centralized",
            "partition": "centralized",
            "seed": int(seed),
            "model_path": str(model_path),
            "validation_report_path": str(
                centralized_report_path
            ),
            "model_sha256": calculate_sha256(
                model_path
            ),
            "validation_report_sha256": (
                calculate_sha256(
                    centralized_report_path
                )
            ),
            "selected_validation_threshold": float(
                threshold
            ),
            "validation_selection_metric": (
                "validation_pr_auc"
            ),
            "selected_checkpoint": {
                "type": "epoch",
                "value": int(
                    seed_report["best_epoch"]
                ),
            },
        }
    )

fedavg_conditions = [
    {
        "condition_id": "fedavg_iid",
        "condition_label": "FedAvg IID",
        "partition": "iid",
        "prefix": "fedavg_iid",
    },
    {
        "condition_id": "fedavg_alpha_05",
        "condition_label": "FedAvg moderate non-IID",
        "partition": "alpha_05",
        "prefix": "fedavg_alpha_05",
    },
    {
        "condition_id": "fedavg_alpha_01",
        "condition_label": "FedAvg severe non-IID",
        "partition": "alpha_01",
        "prefix": "fedavg_alpha_01",
    },
]

for condition in fedavg_conditions:
    for seed in SEEDS:
        experiment_name = (
            f"{condition['prefix']}"
            f"_seed_{seed}"
        )

        report_path = (
            TABLES_DIR
            / f"{experiment_name}_report.json"
        )

        model_path = (
            MODELS_DIR
            / f"{experiment_name}_best.keras"
        )

        if not report_path.exists():
            raise FileNotFoundError(
                report_path
            )

        if not model_path.exists():
            raise FileNotFoundError(
                model_path
            )

        report = load_json(
            report_path
        )

        if int(report["seed"]) != seed:
            raise ValueError(
                f"Seed mismatch in {report_path}."
            )

        if (
            report.get("algorithm")
            not in {
                "FedAvg",
                None,
            }
        ):
            raise ValueError(
                f"Algorithm mismatch in {report_path}."
            )

        observed_partition = report.get(
            "partition_scheme",
            report.get(
                "partition",
                condition["partition"],
            ),
        )

        if (
            str(observed_partition)
            != condition["partition"]
        ):
            raise ValueError(
                f"Partition mismatch in {report_path}."
            )

        require_false(
            report["test_set_used"],
            f"{experiment_name} test_set_used",
        )

        threshold = require_finite_threshold(
            report[
                "selected_validation_threshold"
            ],
            experiment_name,
        )

        records.append(
            {
                "condition_id": (
                    condition["condition_id"]
                ),
                "condition_label": (
                    condition["condition_label"]
                ),
                "algorithm": "FedAvg",
                "partition": (
                    condition["partition"]
                ),
                "seed": int(seed),
                "model_path": str(
                    model_path
                ),
                "validation_report_path": str(
                    report_path
                ),
                "model_sha256": calculate_sha256(
                    model_path
                ),
                "validation_report_sha256": (
                    calculate_sha256(
                        report_path
                    )
                ),
                "selected_validation_threshold": float(
                    threshold
                ),
                "validation_selection_metric": (
                    "validation_pr_auc"
                ),
                "selected_checkpoint": {
                    "type": "round",
                    "value": int(
                        report["best_round"]
                    ),
                },
            }
        )

fedprox_conditions = [
    {
        "condition_id": "fedprox_alpha_05",
        "condition_label": (
            "FedProx moderate non-IID"
        ),
        "partition": "alpha_05",
    },
    {
        "condition_id": "fedprox_alpha_01",
        "condition_label": (
            "FedProx severe non-IID"
        ),
        "partition": "alpha_01",
    },
]

for condition in fedprox_conditions:
    for seed in SEEDS:
        experiment_name = (
            f"fedprox_{condition['partition']}"
            f"_mu_0p01_le_1_seed_{seed}"
        )

        report_path = (
            TABLES_DIR
            / f"{experiment_name}_report.json"
        )

        model_path = (
            MODELS_DIR
            / f"{experiment_name}_best.keras"
        )

        if not report_path.exists():
            raise FileNotFoundError(
                report_path
            )

        if not model_path.exists():
            raise FileNotFoundError(
                model_path
            )

        report = load_json(
            report_path
        )

        if int(report["seed"]) != seed:
            raise ValueError(
                f"Seed mismatch in {report_path}."
            )

        if report["algorithm"] != "FedProx":
            raise ValueError(
                f"Algorithm mismatch in {report_path}."
            )

        if (
            report["partition_scheme"]
            != condition["partition"]
        ):
            raise ValueError(
                f"Partition mismatch in {report_path}."
            )

        if not np.isclose(
            float(report["fedprox_mu"]),
            0.01,
        ):
            raise ValueError(
                f"FedProx mu mismatch in {report_path}."
            )

        if int(report["local_epochs"]) != 1:
            raise ValueError(
                f"Local-epoch mismatch in {report_path}."
            )

        require_false(
            report["test_set_used"],
            f"{experiment_name} test_set_used",
        )

        threshold = require_finite_threshold(
            report[
                "selected_validation_threshold"
            ],
            experiment_name,
        )

        records.append(
            {
                "condition_id": (
                    condition["condition_id"]
                ),
                "condition_label": (
                    condition["condition_label"]
                ),
                "algorithm": "FedProx",
                "partition": (
                    condition["partition"]
                ),
                "seed": int(seed),
                "model_path": str(
                    model_path
                ),
                "validation_report_path": str(
                    report_path
                ),
                "model_sha256": calculate_sha256(
                    model_path
                ),
                "validation_report_sha256": (
                    calculate_sha256(
                        report_path
                    )
                ),
                "selected_validation_threshold": float(
                    threshold
                ),
                "validation_selection_metric": (
                    "validation_pr_auc"
                ),
                "selected_checkpoint": {
                    "type": "round",
                    "value": int(
                        report["best_round"]
                    ),
                },
                "fedprox_mu": 0.01,
                "local_epochs": 1,
            }
        )

expected_condition_ids = {
    "centralized",
    "fedavg_iid",
    "fedavg_alpha_05",
    "fedavg_alpha_01",
    "fedprox_alpha_05",
    "fedprox_alpha_01",
}

observed_condition_ids = {
    record["condition_id"]
    for record in records
}

if observed_condition_ids != expected_condition_ids:
    raise RuntimeError(
        "Frozen condition set is incomplete."
    )

if len(records) != 18:
    raise RuntimeError(
        f"Expected 18 frozen models, "
        f"found {len(records)}."
    )

record_keys = [
    (
        record["condition_id"],
        record["seed"],
    )
    for record in records
]

if len(record_keys) != len(set(record_keys)):
    raise RuntimeError(
        "Duplicate condition-seed records detected."
    )

protocol = {
    "protocol_version": 1,
    "created_at_utc": (
        datetime.now(
            timezone.utc
        ).isoformat()
    ),
    "study": (
        "RSNA pneumonia-associated "
        "lung-opacity classification"
    ),
    "purpose": (
        "Single frozen final test-set evaluation"
    ),
    "test_manifest_path": str(
        TEST_MANIFEST_PATH
    ),
    "test_manifest_opened_or_read": False,
    "test_threshold_tuning_allowed": False,
    "test_model_selection_allowed": False,
    "validation_selection_metric": (
        "validation_pr_auc"
    ),
    "threshold_source": (
        "validation_selected_threshold"
    ),
    "seeds": SEEDS,
    "number_of_frozen_models": int(
        len(records)
    ),
    "conditions": records,
    "final_test_report_path": str(
        FINAL_REPORT_PATH
    ),
    "test_set_used": False,
}

with PROTOCOL_PATH.open(
    "x",
    encoding="utf-8",
) as file:
    json.dump(
        protocol,
        file,
        indent=2,
    )

print(
    "\nFINAL TEST PROTOCOL FROZEN"
)

print("=" * 43)

print(
    "Protocol:",
    PROTOCOL_PATH,
)

print(
    "Test manifest exists:",
    TEST_MANIFEST_PATH.exists(),
)

print(
    "Test manifest opened or read:",
    False,
)

print(
    "Frozen models:",
    len(records),
)

print(
    "Conditions:",
    len(observed_condition_ids),
)

print(
    "Seeds:",
    SEEDS,
)

print(
    "Test threshold tuning allowed:",
    False,
)

print(
    "Test model selection allowed:",
    False,
)

print("\nFROZEN CONDITION-SEED RECORDS")

for record in records:
    print(
        f"{record['condition_id']}, "
        f"seed={record['seed']}, "
        f"checkpoint="
        f"{record['selected_checkpoint']['value']}, "
        f"threshold="
        f"{record['selected_validation_threshold']:.6f}"
    )

print("\nTest set used:", False)
