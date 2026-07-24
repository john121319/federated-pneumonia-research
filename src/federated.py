import numpy as np


def weighted_average_weights(
    client_weight_sets,
    client_sample_counts,
):
    """
    Perform sample-size-weighted FedAvg.

    Every item in client_weight_sets is the complete list
    returned by model.get_weights() for one client.

    Larger clients receive proportionally greater influence.
    """

    if not client_weight_sets:
        raise ValueError(
            "No client weights were provided."
        )

    if len(client_weight_sets) != len(
        client_sample_counts
    ):
        raise ValueError(
            "The number of client weight sets does not "
            "match the number of client sample counts."
        )

    sample_counts = np.asarray(
        client_sample_counts,
        dtype=np.float64,
    )

    if not np.isfinite(sample_counts).all():
        raise ValueError(
            "Client sample counts contain non-finite values."
        )

    if np.any(sample_counts <= 0):
        raise ValueError(
            "Every client must contain at least one sample."
        )

    total_samples = float(
        sample_counts.sum()
    )

    if total_samples <= 0:
        raise ValueError(
            "The total client sample count must be positive."
        )

    number_of_weight_arrays = len(
        client_weight_sets[0]
    )

    for client_index, weight_set in enumerate(
        client_weight_sets
    ):
        if len(weight_set) != number_of_weight_arrays:
            raise ValueError(
                "Client weight-array counts do not match. "
                f"Client index: {client_index}"
            )

    averaged_weights = []

    for weight_index in range(
        number_of_weight_arrays
    ):
        reference_array = np.asarray(
            client_weight_sets[0][weight_index]
        )

        weighted_sum = np.zeros(
            reference_array.shape,
            dtype=np.float64,
        )

        for (
            client_weight_set,
            client_sample_count,
        ) in zip(
            client_weight_sets,
            sample_counts,
        ):
            client_array = np.asarray(
                client_weight_set[weight_index]
            )

            if client_array.shape != reference_array.shape:
                raise ValueError(
                    "Client weight shapes do not match at "
                    f"weight index {weight_index}."
                )

            client_fraction = (
                float(client_sample_count)
                / total_samples
            )

            weighted_sum += (
                client_array.astype(
                    np.float64
                )
                * client_fraction
            )

        averaged_weights.append(
            weighted_sum.astype(
                reference_array.dtype
            )
        )

    return averaged_weights


def weight_l2_distance(
    first_weight_set,
    second_weight_set,
):
    """
    Calculate the L2 distance between two complete
    model-weight sets.
    """

    if len(first_weight_set) != len(
        second_weight_set
    ):
        raise ValueError(
            "Weight-set lengths do not match."
        )

    squared_distance = 0.0

    for weight_index, (
        first_array,
        second_array,
    ) in enumerate(
        zip(
            first_weight_set,
            second_weight_set,
        )
    ):
        first_array = np.asarray(
            first_array
        )

        second_array = np.asarray(
            second_array
        )

        if first_array.shape != second_array.shape:
            raise ValueError(
                "Weight shapes do not match at index "
                f"{weight_index}."
            )

        difference = (
            first_array.astype(
                np.float64
            )
            - second_array.astype(
                np.float64
            )
        )

        squared_distance += float(
            np.sum(
                np.square(
                    difference
                )
            )
        )

    return float(
        np.sqrt(
            squared_distance
        )
    )