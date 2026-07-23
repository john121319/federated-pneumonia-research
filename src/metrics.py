import json
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
def validate_binary_inputs(
    true_labels,
    probabilities,
):

    true_labels = np.asarray(
        true_labels,
        dtype=np.int32,
    ).reshape(-1)


    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(-1)


    if len(true_labels) != len(
        probabilities
    ):

        raise ValueError(
            "Label and probability counts "
            "do not match."
        )


    if len(true_labels) == 0:

        raise ValueError(
            "No samples were provided."
        )


    if not np.isfinite(
        probabilities
    ).all():

        raise ValueError(
            "Probabilities contain "
            "non-finite values."
        )


    observed_labels = set(
        np.unique(
            true_labels
        )
    )


    if not observed_labels.issubset(
        {
            0,
            1,
        }
    ):

        raise ValueError(
            "Labels must contain only 0 and 1."
        )


    if len(
        observed_labels
    ) != 2:

        raise ValueError(
            "Both classes must be present "
            "to calculate ROC-AUC."
        )


    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )


    return (
        true_labels,
        probabilities,
    )
def find_youden_threshold(
    true_labels,
    probabilities,
):

    (
        true_labels,
        probabilities,
    ) = validate_binary_inputs(
        true_labels,
        probabilities,
    )


    (
        false_positive_rates,
        true_positive_rates,
        thresholds,
    ) = roc_curve(
        true_labels,
        probabilities,
    )


    finite_mask = np.isfinite(
        thresholds
    )


    false_positive_rates = (
        false_positive_rates[
            finite_mask
        ]
    )


    true_positive_rates = (
        true_positive_rates[
            finite_mask
        ]
    )


    thresholds = thresholds[
        finite_mask
    ]


    if len(
        thresholds
    ) == 0:

        raise RuntimeError(
            "No finite ROC thresholds "
            "were available."
        )


    youden_scores = (
        true_positive_rates
        - false_positive_rates
    )


    best_index = int(
        np.argmax(
            youden_scores
        )
    )


    selected_threshold = float(
        thresholds[
            best_index
        ]
    )


    return float(
        np.clip(
            selected_threshold,
            0.0,
            1.0,
        )
    )

def calculate_binary_metrics(
    true_labels,
    probabilities,
    threshold,
):

    (
        true_labels,
        probabilities,
    ) = validate_binary_inputs(
        true_labels,
        probabilities,
    )


    threshold = float(
        threshold
    )


    if (
        threshold < 0.0
        or threshold > 1.0
    ):

        raise ValueError(
            "Threshold must be within [0, 1]."
        )


    predicted_labels = (
        probabilities
        >= threshold
    ).astype(
        np.int32
    )


    confusion_values = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[
            0,
            1,
        ],
    )


    (
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    ) = confusion_values.ravel()


    sensitivity = (
        true_positive
        / (
            true_positive
            + false_negative
        )
    )


    specificity = (
        true_negative
        / (
            true_negative
            + false_positive
        )
    )


    negative_predictive_value = (
        true_negative
        / (
            true_negative
            + false_negative
        )
        if (
            true_negative
            + false_negative
        ) > 0
        else 0.0
    )


    precision_values, \
    recall_values, \
    _ = precision_recall_curve(
        true_labels,
        probabilities,
    )


    pr_auc = float(
        auc(
            recall_values,
            precision_values,
        )
    )


    clipped_probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )


    return {
        "threshold": float(
            threshold
        ),

        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),

        "precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),

        "sensitivity": float(
            sensitivity
        ),

        "recall": float(
            recall_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "negative_predictive_value": float(
            negative_predictive_value
        ),

        "f1_score": float(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                true_labels,
                probabilities,
            )
        ),

        "pr_auc": float(
            pr_auc
        ),

        "average_precision": float(
            average_precision_score(
                true_labels,
                probabilities,
            )
        ),

        "log_loss": float(
            log_loss(
                true_labels,
                clipped_probabilities,
                labels=[
                    0,
                    1,
                ],
            )
        ),

        "true_negative": int(
            true_negative
        ),

        "false_positive": int(
            false_positive
        ),

        "false_negative": int(
            false_negative
        ),

        "true_positive": int(
            true_positive
        ),

        "sample_count": int(
            len(
                true_labels
            )
        ),

        "positive_count": int(
            true_labels.sum()
        ),

        "negative_count": int(
            len(
                true_labels
            )
            - true_labels.sum()
        ),
    }
def save_json(
    data,
    output_path,
):

    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )