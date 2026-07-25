import gc
import platform
import sys
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    BATCH_SIZE,
    FEDERATED_ROUNDS,
    FIGURES_DIR,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MANIFEST_DIR,
    MODELS_DIR,
    NUM_CLIENTS,
    PARTITION_DIR,
    RAW_RESULTS_DIR,
    TABLES_DIR,
)

from src.data import (
    calculate_class_weights,
    create_dataset_from_dataframe,
)

from src.federated import (
    weight_l2_distance,
    weighted_average_weights,
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
parser = argparse.ArgumentParser(
    description="Train FedAvg on an IID federated partition."
)

parser.add_argument(
    "--seed",
    type=int,
    required=True,
    choices=[
        11,
        22,
        33,
    ],
)

arguments = parser.parse_args()

SEED = arguments.seed

PARTITION_SCHEME = "iid"

EXPERIMENT_NAME = (
    f"fedavg_{PARTITION_SCHEME}_seed_{SEED}"
)
TRAIN_MANIFEST_PATH = (
    MANIFEST_DIR
    / "train_cached.csv"
)

VALIDATION_MANIFEST_PATH = (
    MANIFEST_DIR
    / "validation_cached.csv"
)

CLIENT_PARTITION_DIRECTORY = (
    PARTITION_DIR
    / f"seed_{SEED}"
    / PARTITION_SCHEME
)
BEST_MODEL_PATH = (
    MODELS_DIR
    / f"{EXPERIMENT_NAME}_best.keras"
)

FINAL_MODEL_PATH = (
    MODELS_DIR
    / f"{EXPERIMENT_NAME}_final.keras"
)

ROUND_HISTORY_PATH = (
    TABLES_DIR
    / f"{EXPERIMENT_NAME}_round_history.csv"
)

CLIENT_HISTORY_PATH = (
    TABLES_DIR
    / f"{EXPERIMENT_NAME}_client_history.csv"
)

REPORT_PATH = (
    TABLES_DIR
    / f"{EXPERIMENT_NAME}_report.json"
)

BEST_PREDICTIONS_PATH = (
    RAW_RESULTS_DIR
    / (
        f"{EXPERIMENT_NAME}"
        "_best_validation_predictions.csv"
    )
)

CONVERGENCE_FIGURE_PATH = (
    FIGURES_DIR
    / f"{EXPERIMENT_NAME}_convergence.png"
)

UPDATE_FIGURE_PATH = (
    FIGURES_DIR
    / f"{EXPERIMENT_NAME}_global_update.png"
)

CONFUSION_FIGURE_PATH = (
    FIGURES_DIR
    / f"{EXPERIMENT_NAME}_confusion_matrix.png"
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
print(
    "\nFEDAVG IID MAIN EXPERIMENT"
)

print(
    "=========================="
)

print(
    "Python:",
    sys.version,
)

print(
    "TensorFlow:",
    tf.__version__,
)

print(
    "Platform:",
    platform.platform(),
)

print(
    "Physical devices:",
    tf.config.list_physical_devices(),
)

print(
    "Algorithm: FedAvg"
)

print(
    "Partition:",
    PARTITION_SCHEME,
)

print(
    "Seed:",
    SEED,
)

print(
    "Clients:",
    NUM_CLIENTS,
)

print(
    "Federated rounds:",
    FEDERATED_ROUNDS,
)

print(
    "Local epochs:",
    LOCAL_EPOCHS,
)

print(
    "Batch size:",
    BATCH_SIZE,
)

print(
    "Learning rate:",
    LEARNING_RATE,
)
try:

    tf.config.experimental.enable_op_determinism()

except Exception as error:

    print(
        "Could not enable deterministic operations:",
        repr(error),
    )
tf.keras.utils.set_random_seed(
    SEED
)

if not TRAIN_MANIFEST_PATH.exists():

    raise FileNotFoundError(
        TRAIN_MANIFEST_PATH
    )


if not VALIDATION_MANIFEST_PATH.exists():

    raise FileNotFoundError(
        VALIDATION_MANIFEST_PATH
    )


if not CLIENT_PARTITION_DIRECTORY.exists():

    raise FileNotFoundError(
        CLIENT_PARTITION_DIRECTORY
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
    "\nDATA"
)

print(
    "===="
)

print(
    "Training images:",
    len(train_dataframe),
)

print(
    "Training patients:",
    train_dataframe[
        "original_patient_id"
    ].nunique(),
)

print(
    "Validation images:",
    len(validation_dataframe),
)

print(
    "Validation patients:",
    validation_dataframe[
        "original_patient_id"
    ].nunique(),
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

training_validation_overlap = len(
    training_patient_ids
    & validation_patient_ids
)

if training_validation_overlap != 0:

    raise RuntimeError(
        "Patient overlap detected between training "
        "and validation data: "
        f"{training_validation_overlap}"
    )


print(
    "Training-validation patient overlap:",
    training_validation_overlap,
)
client_dataframes = {}

all_client_dataframes = []


print(
    "\nCLIENT MANIFESTS"
)

print(
    "================"
)


for client_id in range(
    NUM_CLIENTS
):

    client_path = (
        CLIENT_PARTITION_DIRECTORY
        / f"client_{client_id}.csv"
    )

    if not client_path.exists():

        raise FileNotFoundError(
            client_path
        )

    client_dataframe = pd.read_csv(
        client_path,
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
        "client_id",
    }

    missing_columns = (
        required_columns
        - set(client_dataframe.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Client {client_id} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    observed_client_ids = set(
        client_dataframe[
            "client_id"
        ]
        .astype(int)
        .unique()
    )

    if observed_client_ids != {
        client_id
    }:

        raise ValueError(
            f"Client {client_id} contains incorrect "
            f"client IDs: {sorted(observed_client_ids)}"
        )

    if client_dataframe[
        "exam_id"
    ].duplicated().any():

        raise ValueError(
            f"Client {client_id} contains duplicate "
            "examination IDs."
        )

    client_dataframes[
        client_id
    ] = client_dataframe

    all_client_dataframes.append(
        client_dataframe
    )

    positive_count = int(
        client_dataframe[
            "label"
        ].sum()
    )

    negative_count = int(
        len(client_dataframe)
        - positive_count
    )

    print(
        f"Client {client_id}: "
        f"images={len(client_dataframe)}, "
        f"patients="
        f"{client_dataframe['original_patient_id'].nunique()}, "
        f"positive={positive_count}, "
        f"negative={negative_count}"
    )
combined_clients_dataframe = pd.concat(
    all_client_dataframes,
    ignore_index=True,
)


if len(
    combined_clients_dataframe
) != len(
    train_dataframe
):

    raise RuntimeError(
        "Combined client count does not match "
        "the centralized training manifest."
    )


if combined_clients_dataframe[
    "exam_id"
].duplicated().any():

    raise RuntimeError(
        "An examination was assigned to "
        "multiple clients."
    )


training_exam_ids = set(
    train_dataframe[
        "exam_id"
    ].astype(str)
)


client_exam_ids = set(
    combined_clients_dataframe[
        "exam_id"
    ].astype(str)
)


if training_exam_ids != client_exam_ids:

    raise RuntimeError(
        "Client examination IDs do not match "
        "the training manifest."
    )


patient_client_counts = (
    combined_clients_dataframe
    .groupby(
        "original_patient_id"
    )[
        "client_id"
    ]
    .nunique()
)


patient_overlap_between_clients = int(
    (
        patient_client_counts > 1
    ).sum()
)


if patient_overlap_between_clients != 0:

    raise RuntimeError(
        "Patient overlap detected between clients: "
        f"{patient_overlap_between_clients}"
    )


print(
    "\nPARTITION VALIDATION"
)

print(
    "===================="
)

print(
    "Combined client images:",
    len(combined_clients_dataframe),
)

print(
    "Combined client patients:",
    combined_clients_dataframe[
        "original_patient_id"
    ].nunique(),
)

print(
    "Patient overlap between clients:",
    patient_overlap_between_clients,
)

print(
    "All examinations assigned exactly once:",
    True,
)
class_weights = calculate_class_weights(
    train_dataframe
)


print(
    "\nGLOBAL CLASS WEIGHTS"
)

print(
    "===================="
)

print(
    "Negative:",
    f"{class_weights[0]:.6f}",
)

print(
    "Positive:",
    f"{class_weights[1]:.6f}",
)
validation_dataset = (
    create_dataset_from_dataframe(
        dataframe=validation_dataframe,
        training=False,
        seed=SEED,
        batch_size=BATCH_SIZE,
    )
)


validation_true_labels = (
    validation_dataframe[
        "label"
    ]
    .astype(np.int32)
    .to_numpy()
)
tf.keras.backend.clear_session()

gc.collect()


tf.keras.utils.set_random_seed(
    SEED
)


global_model = build_model(
    use_augmentation=True
)


initial_global_weights = (
    global_model.get_weights()
)
initial_probabilities = (
    global_model.predict(
        validation_dataset,
        verbose=0,
    )
    .reshape(-1)
)


initial_metrics = (
    calculate_binary_metrics(
        true_labels=validation_true_labels,
        probabilities=initial_probabilities,
        threshold=0.5,
    )
)


print(
    "\nROUND 0 — UNTRAINED GLOBAL MODEL"
)

print(
    "================================"
)

print(
    "ROC-AUC:",
    f"{initial_metrics['roc_auc']:.6f}",
)

print(
    "PR-AUC:",
    f"{initial_metrics['pr_auc']:.6f}",
)

print(
    "Accuracy:",
    f"{initial_metrics['accuracy']:.6f}",
)
round_history_rows = [
    {
        "round": 0,
        "roc_auc": float(
            initial_metrics[
                "roc_auc"
            ]
        ),
        "pr_auc": float(
            initial_metrics[
                "pr_auc"
            ]
        ),
        "accuracy_at_0_5": float(
            initial_metrics[
                "accuracy"
            ]
        ),
        "sensitivity_at_0_5": float(
            initial_metrics[
                "sensitivity"
            ]
        ),
        "specificity_at_0_5": float(
            initial_metrics[
                "specificity"
            ]
        ),
        "f1_score_at_0_5": float(
            initial_metrics[
                "f1_score"
            ]
        ),
        "log_loss": float(
            initial_metrics[
                "log_loss"
            ]
        ),
        "global_update_l2": 0.0,
        "mean_client_update_l2": 0.0,
        "minimum_client_update_l2": 0.0,
        "maximum_client_update_l2": 0.0,
        "round_seconds": 0.0,
    }
]


client_history_rows = []


best_validation_pr_auc = float(
    "-inf"
)


best_round = None


total_training_start_time = time.time()
for round_number in range(
    1,
    FEDERATED_ROUNDS + 1,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"FEDERATED ROUND {round_number}"
    )

    print(
        "=" * 60
    )

    round_start_time = time.time()

    global_weights_before_round = (
        global_model.get_weights()
    )

    client_weight_sets = []

    client_sample_counts = []

    round_client_update_values = []
    for client_id in range(
        NUM_CLIENTS
    ):

        client_dataframe = (
            client_dataframes[
                client_id
            ]
        )

        client_sample_count = int(
            len(client_dataframe)
        )

        client_seed = int(
            SEED
            + round_number * 1000
            + client_id
        )

        tf.keras.utils.set_random_seed(
            client_seed
        )

        client_dataset = (
            create_dataset_from_dataframe(
                dataframe=client_dataframe,
                training=True,
                seed=client_seed,
                batch_size=BATCH_SIZE,
            )
        )

        client_model = build_model(
            use_augmentation=True
        )

        compile_model(
            model=client_model,
            learning_rate=LEARNING_RATE,
        )

        # Every client starts this round from the same
        # current global-model parameters.
        client_model.set_weights(
            global_weights_before_round
        )

        client_training_start_time = (
            time.time()
        )

        client_history = client_model.fit(
            client_dataset,
            epochs=LOCAL_EPOCHS,
            class_weight=class_weights,
            verbose=0,
        )

        client_training_seconds = float(
            time.time()
            - client_training_start_time
        )

        client_weights = (
            client_model.get_weights()
        )

        client_update_l2 = (
            weight_l2_distance(
                global_weights_before_round,
                client_weights,
            )
        )

        client_weight_sets.append(
            client_weights
        )

        client_sample_counts.append(
            client_sample_count
        )

        round_client_update_values.append(
            client_update_l2
        )

        final_local_loss = float(
            client_history.history[
                "loss"
            ][-1]
        )

        final_local_roc_auc = float(
            client_history.history[
                "roc_auc"
            ][-1]
        )

        final_local_pr_auc = float(
            client_history.history[
                "pr_auc"
            ][-1]
        )

        client_history_rows.append(
            {
                "round": int(
                    round_number
                ),
                "client_id": int(
                    client_id
                ),
                "client_seed": int(
                    client_seed
                ),
                "images": int(
                    client_sample_count
                ),
                "local_epochs": int(
                    LOCAL_EPOCHS
                ),
                "loss": float(
                    final_local_loss
                ),
                "roc_auc": float(
                    final_local_roc_auc
                ),
                "pr_auc": float(
                    final_local_pr_auc
                ),
                "update_l2": float(
                    client_update_l2
                ),
                "training_seconds": float(
                    client_training_seconds
                ),
            }
        )

        print(
            f"Client {client_id}: "
            f"images={client_sample_count}, "
            f"loss={final_local_loss:.6f}, "
            f"ROC-AUC={final_local_roc_auc:.6f}, "
            f"PR-AUC={final_local_pr_auc:.6f}, "
            f"update_L2={client_update_l2:.6f}, "
            f"time={client_training_seconds:.2f}s"
        )

        del client_model
        del client_dataset

        gc.collect()
    aggregated_weights = (
        weighted_average_weights(
            client_weight_sets=(
                client_weight_sets
            ),
            client_sample_counts=(
                client_sample_counts
            ),
        )
    )


    global_update_l2 = (
        weight_l2_distance(
            global_weights_before_round,
            aggregated_weights,
        )
    )


    global_model.set_weights(
        aggregated_weights
    )

    validation_probabilities = (
        global_model.predict(
            validation_dataset,
            verbose=0,
        )
        .reshape(-1)
    )


    round_metrics = (
        calculate_binary_metrics(
            true_labels=validation_true_labels,
            probabilities=validation_probabilities,
            threshold=0.5,
        )
    )


    round_seconds = float(
        time.time()
        - round_start_time
    )


    mean_client_update_l2 = float(
        np.mean(
            round_client_update_values
        )
    )


    minimum_client_update_l2 = float(
        np.min(
            round_client_update_values
        )
    )


    maximum_client_update_l2 = float(
        np.max(
            round_client_update_values
        )
    )


    round_history_rows.append(
        {
            "round": int(
                round_number
            ),
            "roc_auc": float(
                round_metrics[
                    "roc_auc"
                ]
            ),
            "pr_auc": float(
                round_metrics[
                    "pr_auc"
                ]
            ),
            "accuracy_at_0_5": float(
                round_metrics[
                    "accuracy"
                ]
            ),
            "sensitivity_at_0_5": float(
                round_metrics[
                    "sensitivity"
                ]
            ),
            "specificity_at_0_5": float(
                round_metrics[
                    "specificity"
                ]
            ),
            "f1_score_at_0_5": float(
                round_metrics[
                    "f1_score"
                ]
            ),
            "log_loss": float(
                round_metrics[
                    "log_loss"
                ]
            ),
            "global_update_l2": float(
                global_update_l2
            ),
            "mean_client_update_l2": float(
                mean_client_update_l2
            ),
            "minimum_client_update_l2": float(
                minimum_client_update_l2
            ),
            "maximum_client_update_l2": float(
                maximum_client_update_l2
            ),
            "round_seconds": float(
                round_seconds
            ),
        }
    )

    pd.DataFrame(
        round_history_rows
    ).to_csv(
        ROUND_HISTORY_PATH,
        index=False,
    )


    pd.DataFrame(
        client_history_rows
    ).to_csv(
        CLIENT_HISTORY_PATH,
        index=False,
    )
    if (
        round_metrics[
            "pr_auc"
        ]
        > best_validation_pr_auc
    ):

        best_validation_pr_auc = float(
            round_metrics[
                "pr_auc"
            ]
        )

        best_round = int(
            round_number
        )

        global_model.save(
            BEST_MODEL_PATH
        )

        print(
            "New best global model saved."
        )
    print(
        "\nGLOBAL VALIDATION"
    )

    print(
        "================="
    )

    print(
        "Round:",
        round_number,
    )

    print(
        "ROC-AUC:",
        f"{round_metrics['roc_auc']:.6f}",
    )

    print(
        "PR-AUC:",
        f"{round_metrics['pr_auc']:.6f}",
    )

    print(
        "Accuracy at 0.5:",
        f"{round_metrics['accuracy']:.6f}",
    )

    print(
        "Sensitivity at 0.5:",
        f"{round_metrics['sensitivity']:.6f}",
    )

    print(
        "Specificity at 0.5:",
        f"{round_metrics['specificity']:.6f}",
    )

    print(
        "Global update L2:",
        f"{global_update_l2:.6f}",
    )

    print(
        "Mean client update L2:",
        f"{mean_client_update_l2:.6f}",
    )

    print(
        "Round time:",
        f"{round_seconds:.2f} seconds",
    )
total_training_seconds = float(
    time.time()
    - total_training_start_time
)
global_model.save(
    FINAL_MODEL_PATH
)
if not BEST_MODEL_PATH.exists():

    raise FileNotFoundError(
        BEST_MODEL_PATH
    )


best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH,
    compile=False,
)
best_validation_probabilities = (
    best_model.predict(
        validation_dataset,
        verbose=0,
    )
    .reshape(-1)
)


if len(
    best_validation_probabilities
) != len(
    validation_true_labels
):

    raise RuntimeError(
        "Prediction count does not match "
        "validation label count."
    )


if not np.isfinite(
    best_validation_probabilities
).all():

    raise RuntimeError(
        "Validation predictions contain "
        "non-finite values."
    )
best_metrics_at_half = (
    calculate_binary_metrics(
        true_labels=validation_true_labels,
        probabilities=best_validation_probabilities,
        threshold=0.5,
    )
)
selected_threshold = (
    find_youden_threshold(
        true_labels=validation_true_labels,
        probabilities=best_validation_probabilities,
    )
)


best_metrics_at_selected_threshold = (
    calculate_binary_metrics(
        true_labels=validation_true_labels,
        probabilities=best_validation_probabilities,
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
] = best_validation_probabilities


predictions_dataframe[
    "prediction_threshold_0_5"
] = (
    best_validation_probabilities
    >= 0.5
).astype(
    np.int32
)


predictions_dataframe[
    "prediction_selected_threshold"
] = (
    best_validation_probabilities
    >= selected_threshold
).astype(
    np.int32
)


predictions_dataframe.to_csv(
    BEST_PREDICTIONS_PATH,
    index=False,
)
round_history_dataframe = pd.DataFrame(
    round_history_rows
)


plt.figure(
    figsize=(
        9,
        5,
    )
)


plt.plot(
    round_history_dataframe[
        "round"
    ],
    round_history_dataframe[
        "roc_auc"
    ],
    marker="o",
    label="Validation ROC-AUC",
)


plt.plot(
    round_history_dataframe[
        "round"
    ],
    round_history_dataframe[
        "pr_auc"
    ],
    marker="o",
    label="Validation PR-AUC",
)


plt.axvline(
    best_round,
    linestyle="--",
    label=f"Best round {best_round}",
)


plt.xlabel(
    "Federated round"
)


plt.ylabel(
    "AUC"
)


plt.title(
    f"FedAvg IID validation convergence — seed {SEED}"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    CONVERGENCE_FIGURE_PATH,
    dpi=200,
    bbox_inches="tight",
)


plt.close()
trained_rounds_dataframe = (
    round_history_dataframe[
        round_history_dataframe[
            "round"
        ] > 0
    ]
)


plt.figure(
    figsize=(
        9,
        5,
    )
)


plt.plot(
    trained_rounds_dataframe[
        "round"
    ],
    trained_rounds_dataframe[
        "global_update_l2"
    ],
    marker="o",
    label="Global update L2",
)


plt.plot(
    trained_rounds_dataframe[
        "round"
    ],
    trained_rounds_dataframe[
        "mean_client_update_l2"
    ],
    marker="o",
    label="Mean client update L2",
)


plt.xlabel(
    "Federated round"
)


plt.ylabel(
    "L2 distance"
)


plt.title(
    f"FedAvg IID model updates — seed {SEED}"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    UPDATE_FIGURE_PATH,
    dpi=200,
    bbox_inches="tight",
)


plt.close()

confusion_values = np.array(
    [
        [
            best_metrics_at_selected_threshold[
                "true_negative"
            ],
            best_metrics_at_selected_threshold[
                "false_positive"
            ],
        ],
        [
            best_metrics_at_selected_threshold[
                "false_negative"
            ],
            best_metrics_at_selected_threshold[
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
    "FedAvg IID validation confusion matrix\n"
    f"Seed {SEED}, round {best_round}, "
    f"threshold={selected_threshold:.3f}"
)


plt.tight_layout()


plt.savefig(
    CONFUSION_FIGURE_PATH,
    dpi=200,
    bbox_inches="tight",
)


plt.close()

report = {
    "experiment_name": (
        EXPERIMENT_NAME
    ),

    "experiment_type": (
        "fedavg_main_training"
    ),

    "algorithm": "FedAvg",

    "partition_scheme": (
        PARTITION_SCHEME
    ),

    "seed": int(
        SEED
    ),

    "number_of_clients": int(
        NUM_CLIENTS
    ),

    "federated_rounds": int(
        FEDERATED_ROUNDS
    ),

    "local_epochs": int(
        LOCAL_EPOCHS
    ),

    "batch_size": int(
        BATCH_SIZE
    ),

    "learning_rate": float(
        LEARNING_RATE
    ),

    "training_images": int(
        len(train_dataframe)
    ),

    "validation_images": int(
        len(validation_dataframe)
    ),

    "training_validation_patient_overlap": int(
        training_validation_overlap
    ),

    "patient_overlap_between_clients": int(
        patient_overlap_between_clients
    ),

    "global_class_weights": {
        "negative": float(
            class_weights[0]
        ),
        "positive": float(
            class_weights[1]
        ),
    },

    "aggregation_weighting": (
        "client_training_sample_count"
    ),

    "optimizer_states_aggregated": False,

    "batch_normalization_weights_aggregated": True,

    "test_set_used": False,

    "total_training_seconds": float(
        total_training_seconds
    ),

    "total_training_minutes": float(
        total_training_seconds
        / 60.0
    ),

    "best_round": int(
        best_round
    ),

    "best_validation_pr_auc": float(
        best_validation_pr_auc
    ),

    "selected_validation_threshold": float(
        selected_threshold
    ),

    "initial_metrics_threshold_0_5": (
        initial_metrics
    ),

    "best_metrics_threshold_0_5": (
        best_metrics_at_half
    ),

    "best_metrics_selected_threshold": (
        best_metrics_at_selected_threshold
    ),

    "round_history": (
        round_history_rows
    ),

    "client_history": (
        client_history_rows
    ),
}


save_json(
    report,
    REPORT_PATH,
)

print(
    "\n"
    + "=" * 60
)


print(
    f"FEDAVG IID SEED {SEED} COMPLETED"
)


print(
    "=" * 60
)


print(
    "Best round:",
    best_round,
)


print(
    "Best validation PR-AUC:",
    f"{best_validation_pr_auc:.6f}",
)


print(
    "Best-model ROC-AUC:",
    f"{best_metrics_at_selected_threshold['roc_auc']:.6f}",
)


print(
    "Selected threshold:",
    f"{selected_threshold:.6f}",
)


print(
    "Accuracy:",
    f"{best_metrics_at_selected_threshold['accuracy']:.6f}",
)


print(
    "Sensitivity:",
    f"{best_metrics_at_selected_threshold['sensitivity']:.6f}",
)


print(
    "Specificity:",
    f"{best_metrics_at_selected_threshold['specificity']:.6f}",
)


print(
    "F1-score:",
    f"{best_metrics_at_selected_threshold['f1_score']:.6f}",
)


print(
    "Balanced accuracy:",
    f"{best_metrics_at_selected_threshold['balanced_accuracy']:.6f}",
)


print(
    "Training time:",
    f"{total_training_seconds / 60.0:.2f} minutes",
)


print(
    "Global model changed from initialization:",
    (
        weight_l2_distance(
            initial_global_weights,
            global_model.get_weights(),
        )
        > 0
    ),
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
    "Round history:",
    ROUND_HISTORY_PATH,
)


print(
    "Client history:",
    CLIENT_HISTORY_PATH,
)


print(
    "Report:",
    REPORT_PATH,
)


print(
    "Predictions:",
    BEST_PREDICTIONS_PATH,
)


print(
    "Convergence figure:",
    CONVERGENCE_FIGURE_PATH,
)


print(
    "Update figure:",
    UPDATE_FIGURE_PATH,
)


print(
    "Confusion matrix:",
    CONFUSION_FIGURE_PATH,
)


print(
    "\nTest set used:",
    False,
)