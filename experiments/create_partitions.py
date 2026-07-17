import json

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
    },
)


print(
    "Training images:",
    len(
        training_dataframe
    ),
)


print(
    "Training patients:",
    training_dataframe[
        "original_patient_id"
    ].nunique(),
)


print(
    "Training positives:",
    int(
        training_dataframe[
            "label"
        ].sum()
    ),
)


print(
    "Training negatives:",
    int(
        len(
            training_dataframe
        )
        - training_dataframe[
            "label"
        ].sum()
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


    partition_dataframe[
        "partition_scheme"
    ] = scheme_name


    partition_dataframe[
        "partition_seed"
    ] = seed


    partition_dataframe[
        "dirichlet_alpha"
    ] = (
        np.nan
        if alpha is None
        else float(alpha)
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


        client_dataframe.to_csv(
            output_directory
            / f"client_{client_id}.csv",
            index=False,
        )


    summary_dataframe = (
        summary_dataframe.copy()
    )


    summary_dataframe[
        "partition_scheme"
    ] = scheme_name


    summary_dataframe[
        "partition_seed"
    ] = seed


    summary_dataframe[
        "dirichlet_alpha"
    ] = (
        np.nan
        if alpha is None
        else float(alpha)
    )


    summary_dataframe.to_csv(
        output_directory
        / "summary.csv",
        index=False,
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
            else float(alpha)
        ),

        "number_of_clients": int(
            NUM_CLIENTS
        ),

        "training_images": int(
            len(
                partition_dataframe
            )
        ),

        "training_patients": int(
            partition_dataframe[
                "original_patient_id"
            ].nunique()
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

        "maximum_relative_size_deviation": (
            maximum_size_deviation
        ),

        "minimum_positive_fraction": float(
            summary_dataframe[
                "positive_fraction"
            ].min()
        ),

        "maximum_positive_fraction": float(
            summary_dataframe[
                "positive_fraction"
            ].max()
        ),

        "positive_fraction_range": (
            positive_fraction_range
        ),

        "minimum_AP_fraction": float(
            summary_dataframe[
                "AP_fraction"
            ].min()
        ),

        "maximum_AP_fraction": float(
            summary_dataframe[
                "AP_fraction"
            ].max()
        ),

        "AP_fraction_range": (
            ap_fraction_range
        ),

        "patient_overlap_count": 0,

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


    return metadata


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
                str(seed)
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

        random_seed=seed,

        minimum_client_images=(
            MIN_CLIENT_IMAGES
        ),
    )


    iid_output_directory = (
        PARTITION_DIR
        / f"seed_{seed}"
        / "iid"
    )


    iid_metadata = save_partition(
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

        seed=seed,

        alpha=None,

        objective=(
            iid_objective
        ),

        attempts=1,
    )


    iid_combined_summary = (
        iid_summary.copy()
    )


    iid_combined_summary[
        "partition_scheme"
    ] = "iid"


    iid_combined_summary[
        "partition_seed"
    ] = seed


    all_summary_dataframes.append(
        iid_combined_summary
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

            random_seed=seed,

            minimum_client_images=(
                MIN_CLIENT_IMAGES
            ),
        )


        output_directory = (
            PARTITION_DIR
            / f"seed_{seed}"
            / alpha_name
        )


        metadata = save_partition(
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

            seed=seed,

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


        noniid_combined_summary = (
            noniid_summary.copy()
        )


        noniid_combined_summary[
            "partition_scheme"
        ] = alpha_name


        noniid_combined_summary[
            "partition_seed"
        ] = seed


        all_summary_dataframes.append(
            noniid_combined_summary
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


# Add the numeric alpha after concatenation.
# IID rows receive NaN.
combined_summary[
    "dirichlet_alpha"
] = (
    combined_summary[
        "partition_scheme"
    ].map(
        {
            "alpha_05": 0.5,

            "alpha_01": 0.1,
        }
    )
)


combined_summary_path = (
    TABLES_DIR
    / "federated_partition_summary.csv"
)


combined_summary.to_csv(
    combined_summary_path,
    index=False,
)


combined_metadata_path = (
    TABLES_DIR
    / "federated_partition_metadata.json"
)


with combined_metadata_path.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        all_metadata,
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
    combined_summary_path
)


print(
    combined_metadata_path
)


print(
    "\nFEDERATED CLIENT "
    "PARTITIONING COMPLETED"
)