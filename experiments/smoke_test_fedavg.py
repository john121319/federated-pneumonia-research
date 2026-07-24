import gc
import platform
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    BATCH_SIZE,
    FEDAVG_SMOKE_ROUNDS,
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
    save_json,
)

from src.model import (
    build_model,
    compile_model,
)
SMOKE_SEED = 11
SMOKE_PARTITION_SCHEME = "iid"

EXPERIMENT_NAME = (
    f"fedavg_smoke_"
    f"{SMOKE_PARTITION_SCHEME}_"
    f"seed_{SMOKE_SEED}"
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
    / f"seed_{SMOKE_SEED}"
    / SMOKE_PARTITION_SCHEME
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

FINAL_PREDICTIONS_PATH = (
    RAW_RESULTS_DIR
    / f"{EXPERIMENT_NAME}_validation_predictions.csv"
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
print(
    "\nFEDAVG IID SMOKE TEST"
)

print(
    "====================="
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
    "Seed:",
    SMOKE_SEED,
)

print(
    "Partition:",
    SMOKE_PARTITION_SCHEME,
)

print(
    "Clients:",
    NUM_CLIENTS,
)

print(
    "Federated rounds:",
    FEDAVG_SMOKE_ROUNDS,
)

print(
    "Local epochs:",
    LOCAL_EPOCHS,
)

print(
    "Learning rate:",
    LEARNING_RATE,
)

print(
    "Batch size:",
    BATCH_SIZE,
)
try:
    tf.config.experimental.enable_op_determinism()

except Exception as error:
    print(
        "Could not enable deterministic operations:",
        repr(error),
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
client_dataframes = {}
all_client_dataframes = []

required_client_columns = {
    "exam_id",
    "original_patient_id",
    "label",
    "view_position",
    "cache_path",
    "client_id",
}


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

    missing_columns = (
        required_client_columns
        - set(
            client_dataframe.columns
        )
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
            f"client identifiers: "
            f"{sorted(observed_client_ids)}"
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

    client_positive_count = int(
        client_dataframe[
            "label"
        ].sum()
    )

    print(
        f"Client {client_id}: "
        f"images={len(client_dataframe)}, "
        f"patients="
        f"{client_dataframe['original_patient_id'].nunique()}, "
        f"positive={client_positive_count}, "
        f"negative="
        f"{len(client_dataframe) - client_positive_count}"
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
        "Combined client image count does not match "
        "the centralized training manifest."
    )

if combined_clients_dataframe[
    "exam_id"
].duplicated().any():
    raise RuntimeError(
        "An examination was assigned to multiple clients."
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
        "Client examination IDs do not match the "
        "centralized training manifest."
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

patient_overlap_count = int(
    (
        patient_client_counts > 1
    ).sum()
)

if patient_overlap_count != 0:
    raise RuntimeError(
        "Patient overlap was detected between clients: "
        f"{patient_overlap_count}"
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
    patient_overlap_count,
)

print(
    "All training examinations assigned once:",
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
        seed=SMOKE_SEED,
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
    SMOKE_SEED
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

initial_metrics = calculate_binary_metrics(
    true_labels=validation_true_labels,
    probabilities=initial_probabilities,
    threshold=0.5,
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
        "roc_auc": initial_metrics[
            "roc_auc"
        ],
        "pr_auc": initial_metrics[
            "pr_auc"
        ],
        "accuracy": initial_metrics[
            "accuracy"
        ],
        "sensitivity": initial_metrics[
            "sensitivity"
        ],
        "specificity": initial_metrics[
            "specificity"
        ],
        "f1_score": initial_metrics[
            "f1_score"
        ],
        "log_loss": initial_metrics[
            "log_loss"
        ],
        "global_update_l2": 0.0,
        "round_seconds": 0.0,
    }
]

client_history_rows = []

best_validation_pr_auc = float(
    "-inf"
)

best_round = None
for round_number in range(
    1,
    FEDAVG_SMOKE_ROUNDS + 1,
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
    for client_id in range(
        NUM_CLIENTS
    ):
        client_dataframe = (
            client_dataframes[
                client_id
            ]
        )

        client_sample_count = int(
            len(
                client_dataframe
            )
        )

        client_seed = int(
            SMOKE_SEED
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
        client_model.set_weights(
            global_weights_before_round
        )

        client_training_start = (
            time.time()
        )

        client_history = (
            client_model.fit(
                client_dataset,
                epochs=LOCAL_EPOCHS,
                class_weight=class_weights,
                verbose=0,
            )
        )

        client_training_seconds = (
            time.time()
            - client_training_start
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
                "loss": final_local_loss,
                "roc_auc": final_local_roc_auc,
                "pr_auc": final_local_pr_auc,
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
            true_labels=(
                validation_true_labels
            ),
            probabilities=(
                validation_probabilities
            ),
            threshold=0.5,
        )
    )

    round_seconds = float(
        time.time()
        - round_start_time
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
            "accuracy": float(
                round_metrics[
                    "accuracy"
                ]
            ),
            "sensitivity": float(
                round_metrics[
                    "sensitivity"
                ]
            ),
            "specificity": float(
                round_metrics[
                    "specificity"
                ]
            ),
            "f1_score": float(
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
        "Accuracy:",
        f"{round_metrics['accuracy']:.6f}",
    )

    print(
        "Sensitivity:",
        f"{round_metrics['sensitivity']:.6f}",
    )

    print(
        "Specificity:",
        f"{round_metrics['specificity']:.6f}",
    )

    print(
        "F1-score:",
        f"{round_metrics['f1_score']:.6f}",
    )

    print(
        "Global update L2:",
        f"{global_update_l2:.6f}",
    )

    print(
        "Round time:",
        f"{round_seconds:.2f} seconds",
    )
global_model.save(
    FINAL_MODEL_PATH
)
final_validation_probabilities = (
    global_model.predict(
        validation_dataset,
        verbose=0,
    )
    .reshape(-1)
)

final_metrics = calculate_binary_metrics(
    true_labels=validation_true_labels,
    probabilities=final_validation_probabilities,
    threshold=0.5,
)

prediction_columns = [
    "exam_id",
    "original_patient_id",
    "detailed_class",
    "label",
    "view_position",
]

final_predictions_dataframe = (
    validation_dataframe[
        prediction_columns
    ]
    .copy()
)

final_predictions_dataframe[
    "probability"
] = final_validation_probabilities

final_predictions_dataframe[
    "prediction_threshold_0_5"
] = (
    final_validation_probabilities
    >= 0.5
).astype(
    np.int32
)

final_predictions_dataframe.to_csv(
    FINAL_PREDICTIONS_PATH,
    index=False,
)
report = {
    "experiment_name": (
        EXPERIMENT_NAME
    ),
    "experiment_type": (
        "fedavg_smoke_test"
    ),
    "algorithm": "FedAvg",
    "partition_scheme": (
        SMOKE_PARTITION_SCHEME
    ),
    "seed": int(
        SMOKE_SEED
    ),
    "number_of_clients": int(
        NUM_CLIENTS
    ),
    "federated_rounds": int(
        FEDAVG_SMOKE_ROUNDS
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
        len(
            train_dataframe
        )
    ),
    "validation_images": int(
        len(
            validation_dataframe
        )
    ),
    "patient_overlap_between_clients": int(
        patient_overlap_count
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
    "best_round": (
        best_round
    ),
    "best_validation_pr_auc": float(
        best_validation_pr_auc
    ),
    "initial_metrics_threshold_0_5": (
        initial_metrics
    ),
    "final_metrics_threshold_0_5": (
        final_metrics
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
    "FEDAVG IID SMOKE TEST COMPLETED"
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
    "Initial ROC-AUC:",
    f"{initial_metrics['roc_auc']:.6f}",
)

print(
    "Final ROC-AUC:",
    f"{final_metrics['roc_auc']:.6f}",
)

print(
    "Initial PR-AUC:",
    f"{initial_metrics['pr_auc']:.6f}",
)

print(
    "Final PR-AUC:",
    f"{final_metrics['pr_auc']:.6f}",
)

print(
    "Final global model changed from initialization:",
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
    FINAL_PREDICTIONS_PATH,
)

print(
    "\nTest set used:",
    False,
)