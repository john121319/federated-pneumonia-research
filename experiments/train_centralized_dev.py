import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    BATCH_SIZE,
    DEVELOPMENT_EPOCHS,
    FIGURES_DIR,
    LEARNING_RATE,
    MANIFEST_DIR,
    MODELS_DIR,
    RAW_RESULTS_DIR,
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

SEED = 11


EXPERIMENT_NAME = (
    f"centralized_dev_seed_{SEED}"
)


tf.keras.utils.set_random_seed(
    SEED
)


try:

    tf.config.experimental.enable_op_determinism()


except Exception as error:

    print(
        "Could not enable deterministic "
        "TensorFlow operations:",
        repr(
            error
        ),
    )

TRAIN_MANIFEST_PATH = (
    MANIFEST_DIR
    / "train_cached.csv"
)


VALIDATION_MANIFEST_PATH = (
    MANIFEST_DIR
    / "validation_cached.csv"
)

BEST_MODEL_PATH = (
    MODELS_DIR
    / f"{EXPERIMENT_NAME}_best.keras"
)


FINAL_MODEL_PATH = (
    MODELS_DIR
    / f"{EXPERIMENT_NAME}_final.keras"
)


HISTORY_PATH = (
    TABLES_DIR
    / f"{EXPERIMENT_NAME}_history.csv"
)


METRICS_JSON_PATH = (
    TABLES_DIR
    / f"{EXPERIMENT_NAME}_metrics.json"
)


METRICS_CSV_PATH = (
    TABLES_DIR
    / f"{EXPERIMENT_NAME}_metrics.csv"
)


MODEL_SUMMARY_PATH = (
    TABLES_DIR
    / f"{EXPERIMENT_NAME}_model_summary.txt"
)


PREDICTIONS_PATH = (
    RAW_RESULTS_DIR
    / (
        f"{EXPERIMENT_NAME}"
        "_validation_predictions.csv"
    )
)


LOSS_FIGURE_PATH = (
    FIGURES_DIR
    / f"{EXPERIMENT_NAME}_loss.png"
)


AUC_FIGURE_PATH = (
    FIGURES_DIR
    / f"{EXPERIMENT_NAME}_auc.png"
)


CONFUSION_FIGURE_PATH = (
    FIGURES_DIR
    / (
        f"{EXPERIMENT_NAME}"
        "_confusion_matrix.png"
    )
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

print(
    "\nCENTRALIZED DEVELOPMENT EXPERIMENT"
)


print(
    "=================================="
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
    "Requested epochs:",
    DEVELOPMENT_EPOCHS,
)


print(
    "Batch size:",
    BATCH_SIZE,
)


print(
    "Learning rate:",
    LEARNING_RATE,
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

train_dataset = (
    create_dataset_from_dataframe(
        dataframe=(
            train_dataframe
        ),

        training=True,

        seed=SEED,

        batch_size=BATCH_SIZE,
    )
)


validation_dataset = (
    create_dataset_from_dataframe(
        dataframe=(
            validation_dataframe
        ),

        training=False,

        seed=SEED,

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


print(
    "\nMODEL SUMMARY"
)


print(
    "============="
)


model.summary()


with MODEL_SUMMARY_PATH.open(
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
        filepath=(
            BEST_MODEL_PATH
        ),

        monitor="val_pr_auc",

        mode="max",

        save_best_only=True,

        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",

        mode="max",

        patience=2,

        restore_best_weights=True,

        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",

        mode="max",

        factor=0.5,

        patience=1,

        min_lr=1e-6,

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

    validation_data=(
        validation_dataset
    ),

    epochs=(
        DEVELOPMENT_EPOCHS
    ),

    class_weight=(
        class_weights
    ),

    callbacks=(
        callbacks
    ),

    verbose=1,
)


training_seconds = (
    time.time()
    - training_start_time
)


model.save(
    FINAL_MODEL_PATH
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
    HISTORY_PATH,
    index=False,
)


if "val_pr_auc" not in (
    history_dataframe.columns
):

    raise RuntimeError(
        "Validation PR-AUC was not found "
        "in the training history."
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


best_validation_pr_auc = float(
    history_dataframe.loc[
        best_epoch_index,
        "val_pr_auc",
    ]
)

if not BEST_MODEL_PATH.exists():

    raise FileNotFoundError(
        BEST_MODEL_PATH
    )


best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)
print(
    "\nVALIDATION PREDICTIONS"
)


print(
    "======================"
)


probabilities = (
    best_model.predict(
        validation_dataset,
        verbose=1,
    )
    .reshape(-1)
)


true_labels = (
    validation_dataframe[
        "label"
    ]
    .astype(
        np.int32
    )
    .to_numpy()
)


if len(
    probabilities
) != len(
    true_labels
):

    raise RuntimeError(
        "Prediction count does not match "
        "the validation-label count."
    )
metrics_at_half = (
    calculate_binary_metrics(
        true_labels=(
            true_labels
        ),

        probabilities=(
            probabilities
        ),

        threshold=0.5,
    )
)


selected_threshold = (
    find_youden_threshold(
        true_labels=(
            true_labels
        ),

        probabilities=(
            probabilities
        ),
    )
)


metrics_at_selected_threshold = (
    calculate_binary_metrics(
        true_labels=(
            true_labels
        ),

        probabilities=(
            probabilities
        ),

        threshold=(
            selected_threshold
        ),
    )
)
prediction_columns = [
    "exam_id",
    "original_patient_id",
    "detailed_class",
    "label",
    "view_position",
]


missing_prediction_columns = (
    set(
        prediction_columns
    )
    - set(
        validation_dataframe.columns
    )
)


if missing_prediction_columns:

    raise ValueError(
        "Validation manifest is missing: "
        + str(
            sorted(
                missing_prediction_columns
            )
        )
    )


predictions_dataframe = (
    validation_dataframe[
        prediction_columns
    ]
    .copy()
)


predictions_dataframe[
    "probability"
] = probabilities


predictions_dataframe[
    "prediction_threshold_0_5"
] = (
    probabilities
    >= 0.5
).astype(
    np.int32
)


predictions_dataframe[
    "prediction_selected_threshold"
] = (
    probabilities
    >= selected_threshold
).astype(
    np.int32
)


predictions_dataframe.to_csv(
    PREDICTIONS_PATH,
    index=False,
)
experiment_report = {
    "experiment_name": (
        EXPERIMENT_NAME
    ),

    "experiment_type": (
        "centralized_development"
    ),

    "seed": int(
        SEED
    ),

    "python_version": (
        sys.version
    ),

    "tensorflow_version": (
        tf.__version__
    ),

    "platform": (
        platform.platform()
    ),

    "physical_devices": [
        str(
            device
        )
        for device
        in tf.config.list_physical_devices()
    ],

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

    "requested_epochs": int(
        DEVELOPMENT_EPOCHS
    ),

    "completed_epochs": int(
        len(
            history_dataframe
        )
    ),

    "best_epoch": int(
        best_epoch
    ),

    "best_validation_pr_auc": float(
        best_validation_pr_auc
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

    "learning_rate": float(
        LEARNING_RATE
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
    experiment_report,
    METRICS_JSON_PATH,
)


metrics_dataframe = pd.DataFrame(
    [
        {
            "threshold_type": (
                "fixed_0.5"
            ),

            **metrics_at_half,
        },

        {
            "threshold_type": (
                "validation_youden"
            ),

            **metrics_at_selected_threshold,
        },
    ]
)


metrics_dataframe.to_csv(
    METRICS_CSV_PATH,
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


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Binary cross-entropy"
)


plt.title(
    "Centralized development training loss"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    LOSS_FIGURE_PATH,
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


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "AUC"
)


plt.title(
    "Centralized development AUC"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    AUC_FIGURE_PATH,
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
    "Validation confusion matrix\n"
    f"Threshold = "
    f"{selected_threshold:.3f}"
)


plt.tight_layout()


plt.savefig(
    CONFUSION_FIGURE_PATH,
    dpi=200,
    bbox_inches="tight",
)


plt.close()

print(
    "\nTRAINING SUMMARY"
)


print(
    "================"
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
    "Best validation PR-AUC:",
    f"{best_validation_pr_auc:.6f}",
)


print(
    "Training time:",
    f"{training_seconds / 60.0:.2f} minutes",
)


print(
    "\nVALIDATION METRICS AT 0.5"
)


print(
    "========================="
)


for metric_name, metric_value in (
    metrics_at_half.items()
):

    print(
        f"{metric_name}: "
        f"{metric_value}"
    )


print(
    "\nSELECTED VALIDATION THRESHOLD"
)


print(
    "============================="
)


print(
    f"{selected_threshold:.6f}"
)


print(
    "\nVALIDATION METRICS AT "
    "SELECTED THRESHOLD"
)


print(
    "========================================"
)


for metric_name, metric_value in (
    metrics_at_selected_threshold.items()
):

    print(
        f"{metric_name}: "
        f"{metric_value}"
    )


print(
    "\nGENERATED FILES"
)


print(
    "==============="
)


print(
    "Best model:",
    BEST_MODEL_PATH,
)


print(
    "Final model:",
    FINAL_MODEL_PATH,
)


print(
    "History:",
    HISTORY_PATH,
)


print(
    "Metrics JSON:",
    METRICS_JSON_PATH,
)


print(
    "Metrics CSV:",
    METRICS_CSV_PATH,
)


print(
    "Predictions:",
    PREDICTIONS_PATH,
)


print(
    "Loss figure:",
    LOSS_FIGURE_PATH,
)


print(
    "AUC figure:",
    AUC_FIGURE_PATH,
)


print(
    "Confusion matrix:",
    CONFUSION_FIGURE_PATH,
)


print(
    "\nCENTRALIZED DEVELOPMENT "
    "EXPERIMENT COMPLETED"
)