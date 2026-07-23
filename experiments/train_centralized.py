import gc
import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    BATCH_SIZE,
    CENTRALIZED_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    FIGURES_DIR,
    LEARNING_RATE,
    MANIFEST_DIR,
    MODELS_DIR,
    RAW_RESULTS_DIR,
    SEEDS,
    TABLES_DIR,
)

from src.data import (
    calculate_class_weights,
    create_dataset_from_dataframe,
)

from src.metrics import (
    calculate_binary_metrics,
    find_youden_threshold,
    save_json,
)

from src.model import (
    build_model,
    compile_model,
)
EXPERIMENT_GROUP = "centralized_main"
TRAIN_MANIFEST_PATH = (
    MANIFEST_DIR
    / "train_cached.csv"
)


VALIDATION_MANIFEST_PATH = (
    MANIFEST_DIR
    / "validation_cached.csv"
)
COMBINED_SUMMARY_PATH = (
    TABLES_DIR
    / "centralized_validation_summary.csv"
)


AGGREGATE_SUMMARY_PATH = (
    TABLES_DIR
    / "centralized_validation_aggregate.csv"
)


COMBINED_REPORT_PATH = (
    TABLES_DIR
    / "centralized_validation_report.json"
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RAW_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

try:

    tf.config.experimental.enable_op_determinism()


except Exception as error:

    print(
        "Could not enable deterministic operations:",
        repr(
            error
        ),
    )

print(
    "\nMAIN CENTRALIZED TRAINING"
)


print(
    "========================="
)


print(
    "Training manifest:",
    TRAIN_MANIFEST_PATH,
)


print(
    "Validation manifest:",
    VALIDATION_MANIFEST_PATH,
)


if not TRAIN_MANIFEST_PATH.exists():

    raise FileNotFoundError(
        TRAIN_MANIFEST_PATH
    )


if not VALIDATION_MANIFEST_PATH.exists():

    raise FileNotFoundError(
        VALIDATION_MANIFEST_PATH
    )
train_dataframe = pd.read_csv(
    TRAIN_MANIFEST_PATH,
    dtype={
        "exam_id": str,
        "original_patient_id": str,
        "cache_path": str,
    },
)


validation_dataframe = pd.read_csv(
    VALIDATION_MANIFEST_PATH,
    dtype={
        "exam_id": str,
        "original_patient_id": str,
        "cache_path": str,
    },
)


required_columns = {
    "exam_id",
    "original_patient_id",
    "detailed_class",
    "label",
    "view_position",
    "cache_path",
}


missing_training_columns = (
    required_columns
    - set(
        train_dataframe.columns
    )
)


missing_validation_columns = (
    required_columns
    - set(
        validation_dataframe.columns
    )
)


if missing_training_columns:

    raise ValueError(
        "Training manifest is missing columns: "
        + str(
            sorted(
                missing_training_columns
            )
        )
    )


if missing_validation_columns:

    raise ValueError(
        "Validation manifest is missing columns: "
        + str(
            sorted(
                missing_validation_columns
            )
        )
    )


if train_dataframe[
    "exam_id"
].duplicated().any():

    raise ValueError(
        "Duplicate training examination IDs detected."
    )


if validation_dataframe[
    "exam_id"
].duplicated().any():

    raise ValueError(
        "Duplicate validation examination IDs detected."
    )


training_patient_ids = set(
    train_dataframe[
        "original_patient_id"
    ].astype(str)
)


validation_patient_ids = set(
    validation_dataframe[
        "original_patient_id"
    ].astype(str)
)


patient_overlap_count = len(
    training_patient_ids
    & validation_patient_ids
)


if patient_overlap_count != 0:

    raise RuntimeError(
        "Patient leakage was detected between "
        "training and validation data: "
        f"{patient_overlap_count} patients."
    )


print(
    "Training images:",
    len(
        train_dataframe
    ),
)


print(
    "Validation images:",
    len(
        validation_dataframe
    ),
)


print(
    "Training patients:",
    train_dataframe[
        "original_patient_id"
    ].nunique(),
)


print(
    "Validation patients:",
    validation_dataframe[
        "original_patient_id"
    ].nunique(),
)


print(
    "Training-validation patient overlap:",
    patient_overlap_count,
)


print(
    "Seeds:",
    SEEDS,
)


print(
    "Maximum epochs:",
    CENTRALIZED_EPOCHS,
)


print(
    "Batch size:",
    BATCH_SIZE,
)


print(
    "Fixed learning rate:",
    LEARNING_RATE,
)


print(
    "Early-stopping patience:",
    EARLY_STOPPING_PATIENCE,
)

class_weights = calculate_class_weights(
    train_dataframe
)


print(
    "\nCLASS WEIGHTS"
)


print(
    "============="
)


print(
    "Negative:",
    f"{class_weights[0]:.6f}",
)


print(
    "Positive:",
    f"{class_weights[1]:.6f}",
)

validation_true_labels = (
    validation_dataframe[
        "label"
    ]
    .astype(
        np.int32
    )
    .to_numpy()
)

all_seed_summary_rows = []

all_seed_reports = []

for seed in SEEDS:

    print(
        "\n"
        + "=" * 60
    )


    print(
        f"CENTRALIZED SEED {seed}"
    )


    print(
        "=" * 60
    )
    tf.keras.backend.clear_session()

    gc.collect()

    tf.keras.utils.set_random_seed(
        seed
    )


    experiment_name = (
        f"{EXPERIMENT_GROUP}"
        f"_seed_{seed}"
    )
    best_model_path = (
        MODELS_DIR
        / f"{experiment_name}_best.keras"
    )


    final_model_path = (
        MODELS_DIR
        / f"{experiment_name}_final.keras"
    )


    history_path = (
        TABLES_DIR
        / f"{experiment_name}_history.csv"
    )


    metrics_json_path = (
        TABLES_DIR
        / f"{experiment_name}_metrics.json"
    )


    metrics_csv_path = (
        TABLES_DIR
        / f"{experiment_name}_metrics.csv"
    )


    model_summary_path = (
        TABLES_DIR
        / f"{experiment_name}_model_summary.txt"
    )


    predictions_path = (
        RAW_RESULTS_DIR
        / (
            f"{experiment_name}"
            "_validation_predictions.csv"
        )
    )


    loss_figure_path = (
        FIGURES_DIR
        / f"{experiment_name}_loss.png"
    )


    auc_figure_path = (
        FIGURES_DIR
        / f"{experiment_name}_auc.png"
    )


    confusion_figure_path = (
        FIGURES_DIR
        / (
            f"{experiment_name}"
            "_confusion_matrix.png"
        )
    )
    train_dataset = (
        create_dataset_from_dataframe(
            dataframe=train_dataframe,
            training=True,
            seed=seed,
            batch_size=BATCH_SIZE,
        )
    )


    validation_dataset = (
        create_dataset_from_dataframe(
            dataframe=validation_dataframe,
            training=False,
            seed=seed,
            batch_size=BATCH_SIZE,
        )
    )
    model = build_model(
        use_augmentation=True
    )


    compile_model(
        model=model,
        learning_rate=LEARNING_RATE,
    )


    with model_summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        model.summary(
            print_fn=lambda line: file.write(
                line + "\n"
            )
        )
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    print(
        "\nTRAINING"
    )


    print(
        "========"
    )


    training_start_time = time.time()


    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=CENTRALIZED_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )


    training_seconds = (
        time.time()
        - training_start_time
    )


    # Save the model state at the end of training.
    model.save(
        final_model_path
    )
    history_dataframe = pd.DataFrame(
        history.history
    )


    history_dataframe.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(
                history_dataframe
            )
            + 1,
        ),
    )


    history_dataframe.to_csv(
        history_path,
        index=False,
    )


    if "val_pr_auc" not in (
        history_dataframe.columns
    ):

        raise RuntimeError(
            "Validation PR-AUC is missing "
            "from the training history."
        )


    best_epoch_index = int(
        history_dataframe[
            "val_pr_auc"
        ].idxmax()
    )


    best_epoch = int(
        history_dataframe.loc[
            best_epoch_index,
            "epoch",
        ]
    )


    best_validation_pr_auc_from_history = float(
        history_dataframe.loc[
            best_epoch_index,
            "val_pr_auc",
        ]
    )
    if not best_model_path.exists():

        raise FileNotFoundError(
            best_model_path
        )


    best_model = tf.keras.models.load_model(
        best_model_path
    )
    print(
        "\nVALIDATION PREDICTIONS"
    )


    print(
        "======================"
    )


    validation_probabilities = (
        best_model.predict(
            validation_dataset,
            verbose=1,
        )
        .reshape(-1)
    )


    if len(
        validation_probabilities
    ) != len(
        validation_true_labels
    ):

        raise RuntimeError(
            "Validation prediction count does "
            "not match validation labels."
        )


    if not np.isfinite(
        validation_probabilities
    ).all():

        raise RuntimeError(
            "Validation predictions contain "
            "non-finite values."
        )
    metrics_at_half = (
        calculate_binary_metrics(
            true_labels=validation_true_labels,
            probabilities=validation_probabilities,
            threshold=0.5,
        )
    )
    selected_threshold = (
        find_youden_threshold(
            true_labels=validation_true_labels,
            probabilities=validation_probabilities,
        )
    )


    metrics_at_selected_threshold = (
        calculate_binary_metrics(
            true_labels=validation_true_labels,
            probabilities=validation_probabilities,
            threshold=selected_threshold,
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
        validation_dataframe[
            prediction_columns
        ]
        .copy()
    )


    predictions_dataframe[
        "probability"
    ] = validation_probabilities


    predictions_dataframe[
        "prediction_threshold_0_5"
    ] = (
        validation_probabilities
        >= 0.5
    ).astype(
        np.int32
    )


    predictions_dataframe[
        "prediction_selected_threshold"
    ] = (
        validation_probabilities
        >= selected_threshold
    ).astype(
        np.int32
    )


    predictions_dataframe.to_csv(
        predictions_path,
        index=False,
    )
    seed_report = {
        "experiment_name": experiment_name,
        "experiment_type": (
            "centralized_main_training"
        ),

        "seed": int(
            seed
        ),

        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "platform": platform.platform(),

        "physical_devices": [
            str(
                device
            )
            for device
            in tf.config.list_physical_devices()
        ],

        "training_manifest": str(
            TRAIN_MANIFEST_PATH
        ),

        "validation_manifest": str(
            VALIDATION_MANIFEST_PATH
        ),

        "training_images": int(
            len(
                train_dataframe
            )
        ),

        "validation_images": int(
            len(
                validation_dataframe
            )
        ),

        "training_patients": int(
            train_dataframe[
                "original_patient_id"
            ].nunique()
        ),

        "validation_patients": int(
            validation_dataframe[
                "original_patient_id"
            ].nunique()
        ),

        "training_validation_patient_overlap": int(
            patient_overlap_count
        ),

        "maximum_epochs": int(
            CENTRALIZED_EPOCHS
        ),

        "completed_epochs": int(
            len(
                history_dataframe
            )
        ),

        "best_epoch": int(
            best_epoch
        ),

        "best_validation_pr_auc_from_history": float(
            best_validation_pr_auc_from_history
        ),

        "training_seconds": float(
            training_seconds
        ),

        "training_minutes": float(
            training_seconds
            / 60.0
        ),

        "batch_size": int(
            BATCH_SIZE
        ),

        "fixed_learning_rate": float(
            LEARNING_RATE
        ),

        "early_stopping_patience": int(
            EARLY_STOPPING_PATIENCE
        ),

        "negative_class_weight": float(
            class_weights[0]
        ),

        "positive_class_weight": float(
            class_weights[1]
        ),

        "metrics_threshold_0_5": (
            metrics_at_half
        ),

        "selected_validation_threshold": float(
            selected_threshold
        ),

        "metrics_selected_threshold": (
            metrics_at_selected_threshold
        ),
    }


    save_json(
        seed_report,
        metrics_json_path,
    )
    metrics_dataframe = pd.DataFrame(
        [
            {
                "seed": int(
                    seed
                ),

                "threshold_type": (
                    "fixed_0.5"
                ),

                **metrics_at_half,
            },

            {
                "seed": int(
                    seed
                ),

                "threshold_type": (
                    "validation_youden"
                ),

                **metrics_at_selected_threshold,
            },
        ]
    )


    metrics_dataframe.to_csv(
        metrics_csv_path,
        index=False,
    )
    plt.figure(
        figsize=(
            8,
            5,
        )
    )


    plt.plot(
        history_dataframe[
            "epoch"
        ],

        history_dataframe[
            "loss"
        ],

        marker="o",

        label="Training loss",
    )


    plt.plot(
        history_dataframe[
            "epoch"
        ],

        history_dataframe[
            "val_loss"
        ],

        marker="o",

        label="Validation loss",
    )


    plt.axvline(
        best_epoch,
        linestyle="--",
        label=(
            f"Best epoch {best_epoch}"
        ),
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Binary cross-entropy"
    )


    plt.title(
        f"Centralized training loss — seed {seed}"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        loss_figure_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close()
    plt.figure(
        figsize=(
            8,
            5,
        )
    )


    plt.plot(
        history_dataframe[
            "epoch"
        ],

        history_dataframe[
            "roc_auc"
        ],

        marker="o",

        label="Training ROC-AUC",
    )


    plt.plot(
        history_dataframe[
            "epoch"
        ],

        history_dataframe[
            "val_roc_auc"
        ],

        marker="o",

        label="Validation ROC-AUC",
    )


    plt.plot(
        history_dataframe[
            "epoch"
        ],

        history_dataframe[
            "pr_auc"
        ],

        marker="o",

        label="Training PR-AUC",
    )


    plt.plot(
        history_dataframe[
            "epoch"
        ],

        history_dataframe[
            "val_pr_auc"
        ],

        marker="o",

        label="Validation PR-AUC",
    )


    plt.axvline(
        best_epoch,
        linestyle="--",
        label=(
            f"Best epoch {best_epoch}"
        ),
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "AUC"
    )


    plt.title(
        f"Centralized AUC — seed {seed}"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        auc_figure_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close()
    confusion_values = np.array(
        [
            [
                metrics_at_selected_threshold[
                    "true_negative"
                ],

                metrics_at_selected_threshold[
                    "false_positive"
                ],
            ],

            [
                metrics_at_selected_threshold[
                    "false_negative"
                ],

                metrics_at_selected_threshold[
                    "true_positive"
                ],
            ],
        ]
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
        f"Validation confusion matrix — seed {seed}\n"
        f"Threshold = {selected_threshold:.3f}"
    )


    plt.tight_layout()


    plt.savefig(
        confusion_figure_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close()

    seed_summary_row = {
        "seed": int(
            seed
        ),

        "learning_rate": float(
            LEARNING_RATE
        ),

        "maximum_epochs": int(
            CENTRALIZED_EPOCHS
        ),

        "completed_epochs": int(
            len(
                history_dataframe
            )
        ),

        "best_epoch": int(
            best_epoch
        ),

        "training_minutes": float(
            training_seconds
            / 60.0
        ),

        "selected_threshold": float(
            selected_threshold
        ),

        "accuracy": float(
            metrics_at_selected_threshold[
                "accuracy"
            ]
        ),

        "precision": float(
            metrics_at_selected_threshold[
                "precision"
            ]
        ),

        "sensitivity": float(
            metrics_at_selected_threshold[
                "sensitivity"
            ]
        ),

        "specificity": float(
            metrics_at_selected_threshold[
                "specificity"
            ]
        ),

        "f1_score": float(
            metrics_at_selected_threshold[
                "f1_score"
            ]
        ),

        "balanced_accuracy": float(
            metrics_at_selected_threshold[
                "balanced_accuracy"
            ]
        ),

        "roc_auc": float(
            metrics_at_selected_threshold[
                "roc_auc"
            ]
        ),

        "pr_auc": float(
            metrics_at_selected_threshold[
                "pr_auc"
            ]
        ),

        "average_precision": float(
            metrics_at_selected_threshold[
                "average_precision"
            ]
        ),

        "log_loss": float(
            metrics_at_selected_threshold[
                "log_loss"
            ]
        ),

        "true_negative": int(
            metrics_at_selected_threshold[
                "true_negative"
            ]
        ),

        "false_positive": int(
            metrics_at_selected_threshold[
                "false_positive"
            ]
        ),

        "false_negative": int(
            metrics_at_selected_threshold[
                "false_negative"
            ]
        ),

        "true_positive": int(
            metrics_at_selected_threshold[
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


    all_seed_summary_rows.append(
        seed_summary_row
    )


    all_seed_reports.append(
        seed_report
    )
    print(
        "\nSEED SUMMARY"
    )


    print(
        "============"
    )


    print(
        "Seed:",
        seed,
    )


    print(
        "Completed epochs:",
        len(
            history_dataframe
        ),
    )


    print(
        "Best epoch:",
        best_epoch,
    )


    print(
        "Selected threshold:",
        f"{selected_threshold:.6f}",
    )


    print(
        "ROC-AUC:",
        f"{metrics_at_selected_threshold['roc_auc']:.6f}",
    )


    print(
        "PR-AUC:",
        f"{metrics_at_selected_threshold['pr_auc']:.6f}",
    )


    print(
        "F1-score:",
        f"{metrics_at_selected_threshold['f1_score']:.6f}",
    )


    print(
        "Sensitivity:",
        f"{metrics_at_selected_threshold['sensitivity']:.6f}",
    )


    print(
        "Specificity:",
        f"{metrics_at_selected_threshold['specificity']:.6f}",
    )


    print(
        "Training time:",
        f"{training_seconds / 60.0:.2f} minutes",
    )

combined_summary_dataframe = pd.DataFrame(
    all_seed_summary_rows
)


combined_summary_dataframe.to_csv(
    COMBINED_SUMMARY_PATH,
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

    metric_values = (
        combined_summary_dataframe[
            metric_name
        ]
        .astype(float)
    )


    aggregate_rows.append(
        {
            "metric": metric_name,

            "mean": float(
                metric_values.mean()
            ),

            "standard_deviation": float(
                metric_values.std(
                    ddof=1
                )
            ),

            "minimum": float(
                metric_values.min()
            ),

            "maximum": float(
                metric_values.max()
            ),

            "number_of_seeds": int(
                len(
                    metric_values
                )
            ),
        }
    )


aggregate_dataframe = pd.DataFrame(
    aggregate_rows
)


aggregate_dataframe.to_csv(
    AGGREGATE_SUMMARY_PATH,
    index=False,
)

combined_report = {
    "experiment_group": (
        EXPERIMENT_GROUP
    ),

    "experiment_type": (
        "centralized_main_training"
    ),

    "seeds": [
        int(
            seed
        )
        for seed
        in SEEDS
    ],

    "fixed_learning_rate": float(
        LEARNING_RATE
    ),

    "batch_size": int(
        BATCH_SIZE
    ),

    "maximum_epochs": int(
        CENTRALIZED_EPOCHS
    ),

    "early_stopping_patience": int(
        EARLY_STOPPING_PATIENCE
    ),

    "training_images": int(
        len(
            train_dataframe
        )
    ),

    "validation_images": int(
        len(
            validation_dataframe
        )
    ),

    "training_validation_patient_overlap": int(
        patient_overlap_count
    ),

    "test_set_used": False,

    "seed_reports": (
        all_seed_reports
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
    "\n"
    + "=" * 60
)


print(
    "CENTRALIZED TRAINING COMPLETED"
)


print(
    "=" * 60
)


print(
    "\nSEED RESULTS"
)


print(
    combined_summary_dataframe.to_string(
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
    "\nGENERATED COMBINED FILES"
)


print(
    "========================"
)


print(
    "Seed summary:",
    COMBINED_SUMMARY_PATH,
)


print(
    "Aggregate summary:",
    AGGREGATE_SUMMARY_PATH,
)


print(
    "Combined report:",
    COMBINED_REPORT_PATH,
)


print(
    "\nTest set used:",
    False,
)