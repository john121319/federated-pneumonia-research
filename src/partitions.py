import numpy as np
import pandas as pd


def build_patient_summary(
    dataframe,
):

    required_columns = {
        "exam_id",
        "original_patient_id",
        "label",
        "view_position",
    }


    missing_columns = (
        required_columns
        - set(
            dataframe.columns
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


    patient_summary = (
        dataframe
        .groupby(
            "original_patient_id",
            as_index=False,
        )
        .agg(
            images=(
                "exam_id",
                "size",
            ),

            positive=(
                "label",
                "sum",
            ),

            AP=(
                "view_position",
                lambda values: int(
                    (
                        values == "AP"
                    ).sum()
                ),
            ),

            PA=(
                "view_position",
                lambda values: int(
                    (
                        values == "PA"
                    ).sum()
                ),
            ),
        )
    )


    patient_summary[
        "negative"
    ] = (
        patient_summary[
            "images"
        ]
        - patient_summary[
            "positive"
        ]
    )


    patient_summary[
        "original_patient_id"
    ] = (
        patient_summary[
            "original_patient_id"
        ].astype(str)
    )


    return patient_summary


def normalized_squared_error(
    actual,
    target,
):

    return (
        (
            actual
            - target
        )
        ** 2
    ) / (
        target
        + 1.0
    )

def assign_patients_to_targets(
    patient_summary,
    target_positive,
    target_negative,
    target_ap,
    target_pa,
    random_seed,
):

    number_of_clients = len(
        target_positive
    )


    random_generator = (
        np.random.default_rng(
            random_seed
        )
    )


    patients = (
        patient_summary.copy()
    )

    patients[
        "random_tie_breaker"
    ] = random_generator.random(
        len(patients)
    )

    patients = (
        patients
        .sort_values(
            [
                "images",
                "random_tie_breaker",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    client_positive = np.zeros(
        number_of_clients,
        dtype=np.float64,
    )


    client_negative = np.zeros(
        number_of_clients,
        dtype=np.float64,
    )


    client_ap = np.zeros(
        number_of_clients,
        dtype=np.float64,
    )


    client_pa = np.zeros(
        number_of_clients,
        dtype=np.float64,
    )


    client_patient_counts = (
        np.zeros(
            number_of_clients,
            dtype=np.int64,
        )
    )


    target_total = (
        target_positive
        + target_negative
    )


    assignments = {}

    initial_client_order = (
        np.argsort(
            -target_total
        )
    )


    for patient_number, patient in (
        patients.iterrows()
    ):

        patient_positive = float(
            patient[
                "positive"
            ]
        )


        patient_negative = float(
            patient[
                "negative"
            ]
        )


        patient_ap = float(
            patient[
                "AP"
            ]
        )


        patient_pa = float(
            patient[
                "PA"
            ]
        )


        if (
            patient_number
            < number_of_clients
        ):

            selected_client = int(
                initial_client_order[
                    patient_number
                ]
            )


        else:

            candidate_scores = []


            for client_id in range(
                number_of_clients
            ):

                current_total = (
                    client_positive[
                        client_id
                    ]
                    + client_negative[
                        client_id
                    ]
                )


                new_total = (
                    current_total
                    + patient_positive
                    + patient_negative
                )


                before_error = (
                    normalized_squared_error(
                        client_positive[
                            client_id
                        ],
                        target_positive[
                            client_id
                        ],
                    )

                    + normalized_squared_error(
                        client_negative[
                            client_id
                        ],
                        target_negative[
                            client_id
                        ],
                    )

                    + 1.00
                    * normalized_squared_error(
                        current_total,
                        target_total[
                            client_id
                        ],
                    )

                    + 0.50
                    * normalized_squared_error(
                        client_ap[
                            client_id
                        ],
                        target_ap[
                            client_id
                        ],
                    )

                    + 0.50
                    * normalized_squared_error(
                        client_pa[
                            client_id
                        ],
                        target_pa[
                            client_id
                        ],
                    )
                )


                after_error = (
                    normalized_squared_error(
                        client_positive[
                            client_id
                        ]
                        + patient_positive,
                        target_positive[
                            client_id
                        ],
                    )

                    + normalized_squared_error(
                        client_negative[
                            client_id
                        ]
                        + patient_negative,
                        target_negative[
                            client_id
                        ],
                    )

                    + 1.00
                    * normalized_squared_error(
                        new_total,
                        target_total[
                            client_id
                        ],
                    )

                    + 0.50
                    * normalized_squared_error(
                        client_ap[
                            client_id
                        ]
                        + patient_ap,
                        target_ap[
                            client_id
                        ],
                    )

                    + 0.50
                    * normalized_squared_error(
                        client_pa[
                            client_id
                        ]
                        + patient_pa,
                        target_pa[
                            client_id
                        ],
                    )
                )


                score_change = (
                    after_error
                    - before_error
                )


                # Tiny reproducible tie breaker.
                score_change += float(
                    random_generator.random()
                    * 1e-10
                )


                candidate_scores.append(
                    score_change
                )


            selected_client = int(
                np.argmin(
                    candidate_scores
                )
            )


        patient_id = str(
            patient[
                "original_patient_id"
            ]
        )


        assignments[
            patient_id
        ] = selected_client


        client_positive[
            selected_client
        ] += patient_positive


        client_negative[
            selected_client
        ] += patient_negative


        client_ap[
            selected_client
        ] += patient_ap


        client_pa[
            selected_client
        ] += patient_pa


        client_patient_counts[
            selected_client
        ] += 1


    total_objective = 0.0


    for client_id in range(
        number_of_clients
    ):

        actual_total = (
            client_positive[
                client_id
            ]
            + client_negative[
                client_id
            ]
        )


        total_objective += (
            normalized_squared_error(
                client_positive[
                    client_id
                ],
                target_positive[
                    client_id
                ],
            )

            + normalized_squared_error(
                client_negative[
                    client_id
                ],
                target_negative[
                    client_id
                ],
            )

            + 1.00
            * normalized_squared_error(
                actual_total,
                target_total[
                    client_id
                ],
            )

            + 0.50
            * normalized_squared_error(
                client_ap[
                    client_id
                ],
                target_ap[
                    client_id
                ],
            )

            + 0.50
            * normalized_squared_error(
                client_pa[
                    client_id
                ],
                target_pa[
                    client_id
                ],
            )
        )


    assignment_counts = {
        "positive": (
            client_positive
        ),

        "negative": (
            client_negative
        ),

        "AP": client_ap,

        "PA": client_pa,

        "patients": (
            client_patient_counts
        ),
    }


    return (
        assignments,
        assignment_counts,
        float(
            total_objective
        ),
    )


def summarize_partition(
    partition_dataframe,
    number_of_clients,
):

    summary_rows = []


    for client_id in range(
        number_of_clients
    ):

        client_dataframe = (
            partition_dataframe[
                partition_dataframe[
                    "client_id"
                ]
                == client_id
            ]
        )


        image_count = len(
            client_dataframe
        )


        patient_count = (
            client_dataframe[
                "original_patient_id"
            ].nunique()
        )


        positive_count = int(
            client_dataframe[
                "label"
            ].sum()
        )


        negative_count = int(
            image_count
            - positive_count
        )


        ap_count = int(
            (
                client_dataframe[
                    "view_position"
                ]
                == "AP"
            ).sum()
        )


        pa_count = int(
            (
                client_dataframe[
                    "view_position"
                ]
                == "PA"
            ).sum()
        )


        summary_rows.append(
            {
                "client_id": (
                    client_id
                ),

                "images": (
                    image_count
                ),

                "patients": (
                    patient_count
                ),

                "positive": (
                    positive_count
                ),

                "negative": (
                    negative_count
                ),

                "positive_fraction": (
                    positive_count
                    / image_count
                    if image_count > 0
                    else 0.0
                ),

                "AP": (
                    ap_count
                ),

                "PA": (
                    pa_count
                ),

                "AP_fraction": (
                    ap_count
                    / image_count
                    if image_count > 0
                    else 0.0
                ),
            }
        )


    return pd.DataFrame(
        summary_rows
    )


def verify_partition(
    source_dataframe,
    partition_dataframe,
    number_of_clients,
    minimum_client_images,
):

    if (
        len(partition_dataframe)
        != len(source_dataframe)
    ):

        raise RuntimeError(
            "Partition row count does not match "
            "the training manifest."
        )


    if (
        partition_dataframe[
            "exam_id"
        ].duplicated().any()
    ):

        raise RuntimeError(
            "An examination appears more than "
            "once in the partition."
        )


    source_exam_ids = set(
        source_dataframe[
            "exam_id"
        ].astype(str)
    )


    partition_exam_ids = set(
        partition_dataframe[
            "exam_id"
        ].astype(str)
    )


    if (
        source_exam_ids
        != partition_exam_ids
    ):

        raise RuntimeError(
            "Partition examinations do not "
            "exactly match the training manifest."
        )


    client_values = set(
        partition_dataframe[
            "client_id"
        ].unique()
    )


    expected_client_values = set(
        range(
            number_of_clients
        )
    )


    if (
        client_values
        != expected_client_values
    ):

        raise RuntimeError(
            "Unexpected client identifiers: "
            f"{sorted(client_values)}"
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


    leaking_patients = (
        patient_client_counts[
            patient_client_counts > 1
        ]
    )


    if (
        len(leaking_patients)
        > 0
    ):

        raise RuntimeError(
            "Some patients were assigned "
            "to multiple clients."
        )


    summary = summarize_partition(
        partition_dataframe,
        number_of_clients,
    )


    if (
        summary[
            "images"
        ].min()
        < minimum_client_images
    ):

        raise RuntimeError(
            "At least one client contains fewer "
            f"than {minimum_client_images} images."
        )


    return summary


def create_iid_partition(
    training_dataframe,
    number_of_clients,
    random_seed,
    minimum_client_images,
):

    patient_summary = (
        build_patient_summary(
            training_dataframe
        )
    )


    total_positive = float(
        patient_summary[
            "positive"
        ].sum()
    )


    total_negative = float(
        patient_summary[
            "negative"
        ].sum()
    )


    total_ap = float(
        patient_summary[
            "AP"
        ].sum()
    )


    total_pa = float(
        patient_summary[
            "PA"
        ].sum()
    )


    equal_proportions = np.full(
        number_of_clients,
        1.0
        / number_of_clients,
    )


    target_positive = (
        total_positive
        * equal_proportions
    )


    target_negative = (
        total_negative
        * equal_proportions
    )


    target_ap = (
        total_ap
        * equal_proportions
    )


    target_pa = (
        total_pa
        * equal_proportions
    )


    (
        assignments,
        _,
        objective,
    ) = assign_patients_to_targets(
        patient_summary=(
            patient_summary
        ),

        target_positive=(
            target_positive
        ),

        target_negative=(
            target_negative
        ),

        target_ap=(
            target_ap
        ),

        target_pa=(
            target_pa
        ),

        random_seed=(
            random_seed
        ),
    )


    partition_dataframe = (
        training_dataframe.copy()
    )


    partition_dataframe[
        "original_patient_id"
    ] = (
        partition_dataframe[
            "original_patient_id"
        ].astype(str)
    )


    partition_dataframe[
        "client_id"
    ] = (
        partition_dataframe[
            "original_patient_id"
        ].map(
            assignments
        )
    )


    if (
        partition_dataframe[
            "client_id"
        ].isna().any()
    ):

        raise RuntimeError(
            "Some patients did not receive "
            "a client assignment."
        )


    partition_dataframe[
        "client_id"
    ] = (
        partition_dataframe[
            "client_id"
        ].astype(int)
    )


    summary = verify_partition(
        source_dataframe=(
            training_dataframe
        ),

        partition_dataframe=(
            partition_dataframe
        ),

        number_of_clients=(
            number_of_clients
        ),

        minimum_client_images=(
            minimum_client_images
        ),
    )


    target_dataframe = (
        pd.DataFrame(
            {
                "client_id": range(
                    number_of_clients
                ),

                "target_total": (
                    target_positive
                    + target_negative
                ),

                "target_positive": (
                    target_positive
                ),

                "target_negative": (
                    target_negative
                ),

                "target_positive_fraction": (
                    target_positive
                    / (
                        target_positive
                        + target_negative
                    )
                ),

                "target_AP": (
                    target_ap
                ),

                "target_PA": (
                    target_pa
                ),
            }
        )
    )


    return (
        partition_dataframe,
        summary,
        target_dataframe,
        objective,
    )


def create_dirichlet_partition(
    training_dataframe,
    number_of_clients,
    alpha,
    random_seed,
    minimum_client_images,
    maximum_attempts=5000,
):

    patient_summary = (
        build_patient_summary(
            training_dataframe
        )
    )


    total_images = float(
        patient_summary[
            "images"
        ].sum()
    )


    total_positive = float(
        patient_summary[
            "positive"
        ].sum()
    )


    total_negative = float(
        patient_summary[
            "negative"
        ].sum()
    )


    total_ap = float(
        patient_summary[
            "AP"
        ].sum()
    )


    total_pa = float(
        patient_summary[
            "PA"
        ].sum()
    )

    target_total = np.full(
        number_of_clients,
        total_images
        / number_of_clients,
        dtype=np.float64,
    )

    target_ap = np.full(
        number_of_clients,
        total_ap
        / number_of_clients,
        dtype=np.float64,
    )


    target_pa = np.full(
        number_of_clients,
        total_pa
        / number_of_clients,
        dtype=np.float64,
    )


    mean_client_images = (
        total_images
        / number_of_clients
    )

    if alpha >= 0.5:

        minimum_positive_images = 100

        minimum_negative_images = 200

        minimum_prevalence_range = 0.15

        maximum_prevalence_range = 0.45

        minimum_allowed_prevalence = 0.05

        maximum_allowed_prevalence = 0.60

        require_low_prevalence = False

        require_high_prevalence = False

    else:

        minimum_positive_images = 20

        minimum_negative_images = 50

        minimum_prevalence_range = 0.50

        maximum_prevalence_range = 0.95

        minimum_allowed_prevalence = 0.001

        maximum_allowed_prevalence = 0.98

        require_low_prevalence = True

        require_high_prevalence = True


    for attempt in range(
        maximum_attempts
    ):

        attempt_seed = (
            random_seed
            + attempt
            * 1009
        )


        random_generator = (
            np.random.default_rng(
                attempt_seed
            )
        )

        positive_proportions = (
            random_generator.dirichlet(
                np.full(
                    number_of_clients,
                    alpha,
                    dtype=np.float64,
                )
            )
        )


        target_positive = (
            total_positive
            * positive_proportions
        )

        target_negative = (
            target_total
            - target_positive
        )


        if (
            target_positive.min()
            < minimum_positive_images
        ):

            continue


        if (
            target_negative.min()
            < minimum_negative_images
        ):

            continue


        negative_target_sum_error = abs(
            target_negative.sum()
            - total_negative
        )


        if (
            negative_target_sum_error
            > 1e-6
        ):

            raise RuntimeError(
                "Negative target counts do not "
                "sum to the dataset total."
            )


        target_prevalence = (
            target_positive
            / target_total
        )


        target_prevalence_range = float(
            target_prevalence.max()
            - target_prevalence.min()
        )


        if (
            target_prevalence_range
            < minimum_prevalence_range
        ):

            continue


        if (
            target_prevalence_range
            > maximum_prevalence_range
        ):

            continue


        if (
            target_prevalence.min()
            < minimum_allowed_prevalence
        ):

            continue


        if (
            target_prevalence.max()
            > maximum_allowed_prevalence
        ):

            continue


        if (
            require_low_prevalence
            and target_prevalence.min()
            > 0.05
        ):

            continue


        if (
            require_high_prevalence
            and target_prevalence.max()
            < 0.55
        ):

            continue


        (
            assignments,
            _,
            objective,
        ) = assign_patients_to_targets(
            patient_summary=(
                patient_summary
            ),

            target_positive=(
                target_positive
            ),

            target_negative=(
                target_negative
            ),

            target_ap=(
                target_ap
            ),

            target_pa=(
                target_pa
            ),

            random_seed=(
                attempt_seed
            ),
        )


        partition_dataframe = (
            training_dataframe.copy()
        )


        partition_dataframe[
            "original_patient_id"
        ] = (
            partition_dataframe[
                "original_patient_id"
            ].astype(str)
        )


        partition_dataframe[
            "client_id"
        ] = (
            partition_dataframe[
                "original_patient_id"
            ].map(
                assignments
            )
        )


        if (
            partition_dataframe[
                "client_id"
            ].isna().any()
        ):

            continue


        partition_dataframe[
            "client_id"
        ] = (
            partition_dataframe[
                "client_id"
            ].astype(int)
        )


        try:

            summary = verify_partition(
                source_dataframe=(
                    training_dataframe
                ),

                partition_dataframe=(
                    partition_dataframe
                ),

                number_of_clients=(
                    number_of_clients
                ),

                minimum_client_images=(
                    minimum_client_images
                ),
            )


        except RuntimeError:

            continue


        maximum_size_deviation = float(
            (
                summary[
                    "images"
                ]
                - mean_client_images
            )
            .abs()
            .max()
            / mean_client_images
        )


        if (
            maximum_size_deviation
            > 0.08
        ):

            continue


        ap_fraction_range = float(
            summary[
                "AP_fraction"
            ].max()
            - summary[
                "AP_fraction"
            ].min()
        )


        if (
            ap_fraction_range
            > 0.10
        ):

            continue



        prevalence_minimum = float(
            summary[
                "positive_fraction"
            ].min()
        )


        prevalence_maximum = float(
            summary[
                "positive_fraction"
            ].max()
        )


        prevalence_range = (
            prevalence_maximum
            - prevalence_minimum
        )


        if (
            prevalence_range
            < minimum_prevalence_range
        ):

            continue


        if (
            prevalence_range
            > maximum_prevalence_range
        ):

            continue


        if (
            prevalence_minimum
            < minimum_allowed_prevalence
        ):

            continue


        if (
            prevalence_maximum
            > maximum_allowed_prevalence
        ):

            continue


        if (
            require_low_prevalence
            and prevalence_minimum
            > 0.05
        ):

            continue


        if (
            require_high_prevalence
            and prevalence_maximum
            < 0.55
        ):

            continue


        target_dataframe = (
            pd.DataFrame(
                {
                    "client_id": range(
                        number_of_clients
                    ),

                    "target_total": (
                        target_total
                    ),

                    "target_positive": (
                        target_positive
                    ),

                    "target_negative": (
                        target_negative
                    ),

                    "target_positive_fraction": (
                        target_prevalence
                    ),

                    "target_AP": (
                        target_ap
                    ),

                    "target_PA": (
                        target_pa
                    ),

                    "positive_proportion": (
                        positive_proportions
                    ),
                }
            )
        )


        return (
            partition_dataframe,
            summary,
            target_dataframe,
            objective,
            attempt + 1,
        )


    raise RuntimeError(
        "Could not create a valid "
        "size-controlled Dirichlet partition "
        f"after {maximum_attempts} attempts. "
        f"alpha={alpha}, seed={random_seed}"
    )