import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DIRICHLET_ALPHAS,
    MANIFEST_DIR,
    MIN_CLIENT_IMAGES,
    NUM_CLIENTS,
    PARTITION_DIR,
    PARTITION_SEEDS,
    TABLES_DIR,
)

from src.partitions import (
    create_dirichlet_partition,
    create_iid_partition,
)
TRAIN_MANIFEST_PATH = (
    MANIFEST_DIR
    / "train_cached.csv"
)

COMBINED_SUMMARY_PATH = (
    TABLES_DIR
    / "federated_partition_summary.csv"
)


COMBINED_METADATA_PATH = (
    TABLES_DIR
    / "federated_partition_metadata.json"
)

PARTITION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
print(
    "\nFEDERATED CLIENT PARTITIONS"
)


print(
    "==========================="
)


print(
    "Training manifest:",
    TRAIN_MANIFEST_PATH,
)


print(
    "Manifest exists:",
    TRAIN_MANIFEST_PATH.exists(),
)


if not TRAIN_MANIFEST_PATH.exists():

    raise FileNotFoundError(
        TRAIN_MANIFEST_PATH
    )


training_dataframe = pd.read_csv(
    TRAIN_MANIFEST_PATH,
    dtype={
        "exam_id": str,
        "original_patient_id": str,
        "cache_path": str,
    },
)


required_columns = {
    "exam_id",
    "original_patient_id",
    "label",
    "view_position",
    "cache_path",
}


missing_columns = (
    required_columns
    - set(
        training_dataframe.columns
    )
)


if missing_columns:

    raise ValueError(
        "Training manifest is missing columns: "
        + str(
            sorted(
                missing_columns
            )
        )
    )


if len(
    training_dataframe
) == 0:

    raise ValueError(
        "Training manifest is empty."
    )


if training_dataframe[
    "exam_id"
].isna().any():

    raise ValueError(
        "Training manifest contains missing "
        "examination identifiers."
    )


if training_dataframe[
    "exam_id"
].duplicated().any():

    duplicate_count = int(
        training_dataframe[
            "exam_id"
        ].duplicated().sum()
    )


    raise ValueError(
        "Training manifest contains duplicate "
        f"examination identifiers: {duplicate_count}"
    )


if training_dataframe[
    "original_patient_id"
].isna().any():

    raise ValueError(
        "Training manifest contains missing "
        "patient identifiers."
    )


empty_patient_ids = (
    training_dataframe[
        "original_patient_id"
    ]
    .astype(str)
    .str.strip()
    .eq("")
)


if empty_patient_ids.any():

    raise ValueError(
        "Training manifest contains empty "
        "patient identifiers."
    )


valid_labels = {
    0,
    1,
}


observed_labels = set(
    training_dataframe[
        "label"
    ]
    .dropna()
    .astype(int)
    .unique()
)


if not observed_labels.issubset(
    valid_labels
):

    raise ValueError(
        "Training labels must be binary 0 or 1. "
        f"Observed values: {sorted(observed_labels)}"
    )


if training_dataframe[
    "label"
].isna().any():

    raise ValueError(
        "Training manifest contains missing labels."
    )


valid_views = {
    "AP",
    "PA",
}


observed_views = set(
    training_dataframe[
        "view_position"
    ]
    .dropna()
    .astype(str)
    .unique()
)


unexpected_views = (
    observed_views
    - valid_views
)


if unexpected_views:

    raise ValueError(
        "Unexpected view positions were found: "
        f"{sorted(unexpected_views)}"
    )


missing_cache_paths = []


for cache_path in (
    training_dataframe[
        "cache_path"
    ].astype(str)
):

    if not Path(
        cache_path
    ).exists():

        missing_cache_paths.append(
            cache_path
        )


if missing_cache_paths:

    raise FileNotFoundError(
        "Some training cache files are missing. "
        f"Missing count: {len(missing_cache_paths)}. "
        f"First missing file: "
        f"{missing_cache_paths[0]}"
    )


training_images = int(
    len(
        training_dataframe
    )
)


training_patients = int(
    training_dataframe[
        "original_patient_id"
    ].nunique()
)


training_positives = int(
    training_dataframe[
        "label"
    ].sum()
)


training_negatives = int(
    training_images
    - training_positives
)


training_ap = int(
    (
        training_dataframe[
            "view_position"
        ]
        == "AP"
    ).sum()
)


training_pa = int(
    (
        training_dataframe[
            "view_position"
        ]
        == "PA"
    ).sum()
)


print(
    "Training images:",
    training_images,
)


print(
    "Training patients:",
    training_patients,
)


print(
    "Training positives:",
    training_positives,
)


print(
    "Training negatives:",
    training_negatives,
)


print(
    "Training AP images:",
    training_ap,
)


print(
    "Training PA images:",
    training_pa,
)


print(
    "Cache paths verified:",
    len(
        training_dataframe
    ),
)

def save_partition(
    partition_dataframe,
    summary_dataframe,
    target_dataframe,
    output_directory,
    scheme_name,
    seed,
    alpha,
    objective,
    attempts,
):

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    partition_dataframe = (
        partition_dataframe.copy()
    )


    summary_dataframe = (
        summary_dataframe.copy()
    )


    target_dataframe = (
        target_dataframe.copy()
    )
    required_partition_columns = {
        "exam_id",
        "original_patient_id",
        "label",
        "view_position",
        "client_id",
    }


    missing_partition_columns = (
        required_partition_columns
        - set(
            partition_dataframe.columns
        )
    )


    if missing_partition_columns:

        raise ValueError(
            "Partition is missing columns: "
            + str(
                sorted(
                    missing_partition_columns
                )
            )
        )
    if len(
        partition_dataframe
    ) != training_images:

        raise RuntimeError(
            f"{scheme_name}, seed {seed}: "
            "partition image count does not match "
            "the training manifest."
        )


    if partition_dataframe[
        "exam_id"
    ].duplicated().any():

        raise RuntimeError(
            f"{scheme_name}, seed {seed}: "
            "duplicate examination assignments "
            "were detected."
        )


    source_exam_ids = set(
        training_dataframe[
            "exam_id"
        ].astype(str)
    )


    partition_exam_ids = set(
        partition_dataframe[
            "exam_id"
        ].astype(str)
    )


    if source_exam_ids != partition_exam_ids:

        missing_exam_count = len(
            source_exam_ids
            - partition_exam_ids
        )


        extra_exam_count = len(
            partition_exam_ids
            - source_exam_ids
        )


        raise RuntimeError(
            f"{scheme_name}, seed {seed}: "
            "partition examination IDs do not "
            "match the training manifest. "
            f"Missing={missing_exam_count}, "
            f"extra={extra_exam_count}"
        )
    observed_client_ids = set(
        partition_dataframe[
            "client_id"
        ]
        .astype(int)
        .unique()
    )


    expected_client_ids = set(
        range(
            NUM_CLIENTS
        )
    )


    if observed_client_ids != expected_client_ids:

        raise RuntimeError(
            f"{scheme_name}, seed {seed}: "
            "unexpected client IDs. "
            f"Expected={sorted(expected_client_ids)}, "
            f"observed={sorted(observed_client_ids)}"
        )
    patient_client_counts = (
        partition_dataframe
        .groupby(
            "original_patient_id"
        )[
            "client_id"
        ]
        .nunique()
    )


    patient_overlap_count = int(
        (
            patient_client_counts > 1
        ).sum()
    )


    if patient_overlap_count != 0:

        raise RuntimeError(
            f"{scheme_name}, seed {seed}: "
            "patient leakage detected. "
            f"{patient_overlap_count} patients "
            "were assigned to multiple clients."
        )
    partition_positives = int(
        partition_dataframe[
            "label"
        ].sum()
    )


    partition_negatives = int(
        len(
            partition_dataframe
        )
        - partition_positives
    )


    partition_patients = int(
        partition_dataframe[
            "original_patient_id"
        ].nunique()
    )


    partition_ap = int(
        (
            partition_dataframe[
                "view_position"
            ]
            == "AP"
        ).sum()
    )


    partition_pa = int(
        (
            partition_dataframe[
                "view_position"
            ]
            == "PA"
        ).sum()
    )


    if partition_positives != training_positives:

        raise RuntimeError(
            f"{scheme_name}, seed {seed}: "
            "positive-count mismatch."
        )


    if partition_negatives != training_negatives:

        raise RuntimeError(
            f"{scheme_name}, seed {seed}: "
            "negative-count mismatch."
        )


    if partition_patients != training_patients:

        raise RuntimeError(
            f"{scheme_name}, seed {seed}: "
            "patient-count mismatch."
        )


    if partition_ap != training_ap:

        raise RuntimeError(
            f"{scheme_name}, seed {seed}: "
            "AP-count mismatch."
        )


    if partition_pa != training_pa:

        raise RuntimeError(
            f"{scheme_name}, seed {seed}: "
            "PA-count mismatch."
        )
    partition_dataframe[
        "partition_scheme"
    ] = scheme_name


    partition_dataframe[
        "partition_seed"
    ] = int(
        seed
    )


    partition_dataframe[
        "dirichlet_alpha"
    ] = (
        np.nan
        if alpha is None
        else float(
            alpha
        )
    )
    partition_dataframe.to_csv(
        output_directory
        / "all_clients.csv",
        index=False,
    )
    for client_id in range(
        NUM_CLIENTS
    ):

        client_dataframe = (
            partition_dataframe[
                partition_dataframe[
                    "client_id"
                ]
                == client_id
            ]
            .copy()
        )


        if len(
            client_dataframe
        ) < MIN_CLIENT_IMAGES:

            raise RuntimeError(
                f"{scheme_name}, seed {seed}, "
                f"client {client_id}: only "
                f"{len(client_dataframe)} images."
            )


        client_dataframe.to_csv(
            output_directory
            / f"client_{client_id}.csv",
            index=False,
        )
    summary_dataframe[
        "partition_scheme"
    ] = scheme_name


    summary_dataframe[
        "partition_seed"
    ] = int(
        seed
    )


    summary_dataframe[
        "dirichlet_alpha"
    ] = (
        np.nan
        if alpha is None
        else float(
            alpha
        )
    )


    summary_dataframe.to_csv(
        output_directory
        / "summary.csv",
        index=False,
    )
    target_dataframe[
        "partition_scheme"
    ] = scheme_name


    target_dataframe[
        "partition_seed"
    ] = int(
        seed
    )


    target_dataframe[
        "dirichlet_alpha"
    ] = (
        np.nan
        if alpha is None
        else float(
            alpha
        )
    )


    target_dataframe.to_csv(
        output_directory
        / "targets.csv",
        index=False,
    )
    mean_client_images = float(
        summary_dataframe[
            "images"
        ].mean()
    )


    maximum_size_deviation = float(
        (
            summary_dataframe[
                "images"
            ]
            - mean_client_images
        )
        .abs()
        .max()
        / mean_client_images
    )


    ap_fraction_range = float(
        summary_dataframe[
            "AP_fraction"
        ].max()
        - summary_dataframe[
            "AP_fraction"
        ].min()
    )


    positive_fraction_range = float(
        summary_dataframe[
            "positive_fraction"
        ].max()
        - summary_dataframe[
            "positive_fraction"
        ].min()
    )


    minimum_positive_fraction = float(
        summary_dataframe[
            "positive_fraction"
        ].min()
    )


    maximum_positive_fraction = float(
        summary_dataframe[
            "positive_fraction"
        ].max()
    )


    minimum_ap_fraction = float(
        summary_dataframe[
            "AP_fraction"
        ].min()
    )


    maximum_ap_fraction = float(
        summary_dataframe[
            "AP_fraction"
        ].max()
    )
    metadata = {
        "scheme": (
            scheme_name
        ),

        "seed": int(
            seed
        ),

        "alpha": (
            None
            if alpha is None
            else float(
                alpha
            )
        ),

        "number_of_clients": int(
            NUM_CLIENTS
        ),

        "source_manifest": str(
            TRAIN_MANIFEST_PATH
        ),

        "training_images": int(
            training_images
        ),

        "training_patients": int(
            training_patients
        ),

        "training_positive_images": int(
            training_positives
        ),

        "training_negative_images": int(
            training_negatives
        ),

        "training_AP_images": int(
            training_ap
        ),

        "training_PA_images": int(
            training_pa
        ),

        "minimum_client_images": int(
            summary_dataframe[
                "images"
            ].min()
        ),

        "maximum_client_images": int(
            summary_dataframe[
                "images"
            ].max()
        ),

        "mean_client_images": float(
            mean_client_images
        ),

        "maximum_relative_size_deviation": float(
            maximum_size_deviation
        ),

        "minimum_positive_fraction": float(
            minimum_positive_fraction
        ),

        "maximum_positive_fraction": float(
            maximum_positive_fraction
        ),

        "positive_fraction_range": float(
            positive_fraction_range
        ),

        "minimum_AP_fraction": float(
            minimum_ap_fraction
        ),

        "maximum_AP_fraction": float(
            maximum_ap_fraction
        ),

        "AP_fraction_range": float(
            ap_fraction_range
        ),

        "patient_overlap_count": int(
            patient_overlap_count
        ),

        "all_training_examinations_assigned_once": True,

        "assignment_objective": float(
            objective
        ),

        "generation_attempts": int(
            attempts
        ),
    }


    with (
        output_directory
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


    return (
        metadata,
        summary_dataframe,
    )
all_summary_dataframes = []

all_metadata = []


for seed in PARTITION_SEEDS:

    print(
        f"\nSEED {seed}"
    )


    print(
        "="
        * (
            len(
                str(
                    seed
                )
            )
            + 5
        )
    )

    (
        iid_partition,
        iid_summary,
        iid_targets,
        iid_objective,
    ) = create_iid_partition(
        training_dataframe=(
            training_dataframe
        ),

        number_of_clients=(
            NUM_CLIENTS
        ),

        random_seed=(
            seed
        ),

        minimum_client_images=(
            MIN_CLIENT_IMAGES
        ),
    )


    iid_output_directory = (
        PARTITION_DIR
        / f"seed_{seed}"
        / "iid"
    )


    (
        iid_metadata,
        iid_saved_summary,
    ) = save_partition(
        partition_dataframe=(
            iid_partition
        ),

        summary_dataframe=(
            iid_summary
        ),

        target_dataframe=(
            iid_targets
        ),

        output_directory=(
            iid_output_directory
        ),

        scheme_name="iid",

        seed=(
            seed
        ),

        alpha=None,

        objective=(
            iid_objective
        ),

        attempts=1,
    )


    all_summary_dataframes.append(
        iid_saved_summary
    )


    all_metadata.append(
        iid_metadata
    )


    print(
        "\nIID"
    )


    print(
        iid_summary.to_string(
            index=False
        )
    )
    for (
        alpha_name,
        alpha_value,
    ) in DIRICHLET_ALPHAS.items():

        (
            noniid_partition,
            noniid_summary,
            noniid_targets,
            noniid_objective,
            attempts,
        ) = create_dirichlet_partition(
            training_dataframe=(
                training_dataframe
            ),

            number_of_clients=(
                NUM_CLIENTS
            ),

            alpha=(
                alpha_value
            ),

            random_seed=(
                seed
            ),

            minimum_client_images=(
                MIN_CLIENT_IMAGES
            ),
        )


        output_directory = (
            PARTITION_DIR
            / f"seed_{seed}"
            / alpha_name
        )


        (
            metadata,
            saved_summary,
        ) = save_partition(
            partition_dataframe=(
                noniid_partition
            ),

            summary_dataframe=(
                noniid_summary
            ),

            target_dataframe=(
                noniid_targets
            ),

            output_directory=(
                output_directory
            ),

            scheme_name=(
                alpha_name
            ),

            seed=(
                seed
            ),

            alpha=(
                alpha_value
            ),

            objective=(
                noniid_objective
            ),

            attempts=(
                attempts
            ),
        )


        all_summary_dataframes.append(
            saved_summary
        )


        all_metadata.append(
            metadata
        )


        print(
            f"\n{alpha_name} "
            f"(alpha={alpha_value}, "
            f"attempts={attempts})"
        )


        print(
            noniid_summary.to_string(
                index=False
            )
        )
combined_summary = pd.concat(
    all_summary_dataframes,
    ignore_index=True,
)


combined_summary.to_csv(
    COMBINED_SUMMARY_PATH,
    index=False,
)

combined_metadata = {
    "research_stage": (
        "federated_client_partitioning"
    ),

    "source_manifest": str(
        TRAIN_MANIFEST_PATH
    ),

    "number_of_clients": int(
        NUM_CLIENTS
    ),

    "partition_seeds": [
        int(
            seed
        )
        for seed
        in PARTITION_SEEDS
    ],

    "dirichlet_alphas": {
        str(
            name
        ): float(
            value
        )
        for name, value
        in DIRICHLET_ALPHAS.items()
    },

    "training_images": int(
        training_images
    ),

    "training_patients": int(
        training_patients
    ),

    "training_positive_images": int(
        training_positives
    ),

    "training_negative_images": int(
        training_negatives
    ),

    "training_AP_images": int(
        training_ap
    ),

    "training_PA_images": int(
        training_pa
    ),

    "partition_count": int(
        len(
            all_metadata
        )
    ),

    "partitions": (
        all_metadata
    ),

    "all_partitions_completed_successfully": True,
}


with COMBINED_METADATA_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        combined_metadata,
        file,
        indent=2,
    )
print(
    "\nGENERATED SUMMARY FILES"
)


print(
    "======================="
)


print(
    COMBINED_SUMMARY_PATH
)


print(
    COMBINED_METADATA_PATH
)


print(
    "\nPARTITION VALIDATION"
)


print(
    "===================="
)


print(
    "Partition conditions:",
    len(
        all_metadata
    ),
)


print(
    "Images per condition:",
    training_images,
)


print(
    "Patients per condition:",
    training_patients,
)


print(
    "Patient overlap in every condition:",
    0,
)


print(
    "All examinations assigned exactly once:",
    True,
)


print(
    "\nFEDERATED CLIENT "
    "PARTITIONING COMPLETED"
)