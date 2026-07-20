import json
import platform
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MANIFEST_DIR,
    TABLES_DIR,
)

from src.data import (
    calculate_class_weights,
    create_dataset_from_dataframe,
)

from src.model import (
    build_model,
    compile_model,
)


SEED = 11


tf.keras.utils.set_random_seed(
    SEED
)


try:

    tf.config.experimental.enable_op_determinism()


except Exception as error:

    print(
        "Deterministic operations "
        "could not be enabled:",
        repr(error),
    )


TRAIN_MANIFEST_PATH = (
    MANIFEST_DIR
    / "train_cached.csv"
)


VALIDATION_MANIFEST_PATH = (
    MANIFEST_DIR
    / "validation_cached.csv"
)


REPORT_PATH = (
    TABLES_DIR
    / "training_pipeline_smoke_test.json"
)


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("\nENVIRONMENT")
print("===========")

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
    "Processor:",
    platform.processor(),
)

print(
    "Physical devices:",
    tf.config.list_physical_devices(),
)


print("\nMANIFESTS")
print("=========")

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
    TRAIN_MANIFEST_PATH
)


validation_dataframe = pd.read_csv(
    VALIDATION_MANIFEST_PATH
)


print(
    "Training images:",
    len(train_dataframe),
)

print(
    "Validation images:",
    len(validation_dataframe),
)

print(
    "Training positives:",
    int(
        train_dataframe[
            "label"
        ].sum()
    ),
)

print(
    "Training negatives:",
    int(
        len(train_dataframe)
        - train_dataframe[
            "label"
        ].sum()
    ),
)



class_weights = (
    calculate_class_weights(
        train_dataframe
    )
)


print("\nCLASS WEIGHTS")
print("=============")

print(
    "Negative class weight:",
    f"{class_weights[0]:.6f}",
)

print(
    "Positive class weight:",
    f"{class_weights[1]:.6f}",
)


train_dataset = (
    create_dataset_from_dataframe(
        dataframe=train_dataframe,
        training=True,
        seed=SEED,
        batch_size=BATCH_SIZE,
    )
)


validation_dataset = (
    create_dataset_from_dataframe(
        dataframe=validation_dataframe,
        training=False,
        seed=SEED,
        batch_size=BATCH_SIZE,
    )
)


training_images, training_labels = next(
    iter(
        train_dataset
    )
)


print("\nTRAINING BATCH")
print("==============")

print(
    "Image shape:",
    training_images.shape,
)

print(
    "Image dtype:",
    training_images.dtype,
)

print(
    "Label shape:",
    training_labels.shape,
)

print(
    "Label dtype:",
    training_labels.dtype,
)

print(
    "Minimum pixel:",
    float(
        tf.reduce_min(
            training_images
        ).numpy()
    ),
)

print(
    "Maximum pixel:",
    float(
        tf.reduce_max(
            training_images
        ).numpy()
    ),
)

print(
    "Mean pixel:",
    float(
        tf.reduce_mean(
            training_images
        ).numpy()
    ),
)

print(
    "Positive labels in batch:",
    int(
        tf.reduce_sum(
            training_labels
        ).numpy()
    ),
)

print(
    "Negative labels in batch:",
    int(
        len(training_labels)
        - tf.reduce_sum(
            training_labels
        ).numpy()
    ),
)


expected_image_shape = (
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    1,
)


if tuple(
    training_images.shape
) != expected_image_shape:

    raise RuntimeError(
        "Unexpected training batch shape: "
        f"{training_images.shape}"
    )


if not bool(
    tf.reduce_all(
        tf.math.is_finite(
            training_images
        )
    ).numpy()
):

    raise RuntimeError(
        "Training images contain "
        "non-finite values."
    )


if float(
    tf.reduce_min(
        training_images
    ).numpy()
) < 0.0:

    raise RuntimeError(
        "Training images contain "
        "values below zero."
    )


if float(
    tf.reduce_max(
        training_images
    ).numpy()
) > 1.0:

    raise RuntimeError(
        "Training images contain "
        "values above one."
    )


model = build_model(
    use_augmentation=True
)


compile_model(
    model
)


print("\nMODEL SUMMARY")
print("=============")

model.summary()


predictions_before_training = model(
    training_images,
    training=False,
)


print("\nFORWARD PASS")
print("============")

print(
    "Prediction shape:",
    predictions_before_training.shape,
)

print(
    "Minimum prediction:",
    float(
        tf.reduce_min(
            predictions_before_training
        ).numpy()
    ),
)

print(
    "Maximum prediction:",
    float(
        tf.reduce_max(
            predictions_before_training
        ).numpy()
    ),
)

print(
    "Mean prediction:",
    float(
        tf.reduce_mean(
            predictions_before_training
        ).numpy()
    ),
)


training_result = model.train_on_batch(
    training_images,
    training_labels,
    class_weight=class_weights,
    return_dict=True,
)


print("\nONE TRAINING STEP")
print("=================")

for metric_name, metric_value in (
    training_result.items()
):

    print(
        f"{metric_name}: "
        f"{float(metric_value):.6f}"
    )


validation_images, validation_labels = next(
    iter(
        validation_dataset
    )
)


validation_result = (
    model.test_on_batch(
        validation_images,
        validation_labels,
        return_dict=True,
    )
)


print("\nONE VALIDATION STEP")
print("===================")

for metric_name, metric_value in (
    validation_result.items()
):

    print(
        f"{metric_name}: "
        f"{float(metric_value):.6f}"
    )


predictions_after_training = model(
    training_images,
    training=False,
)


prediction_change = float(
    tf.reduce_mean(
        tf.abs(
            predictions_after_training
            - predictions_before_training
        )
    ).numpy()
)


print("\nBACKPROPAGATION CHECK")
print("=====================")

print(
    "Mean prediction change:",
    f"{prediction_change:.8f}",
)


if prediction_change <= 0.0:

    raise RuntimeError(
        "Model predictions did not change "
        "after the training step."
    )


report = {
    "seed": int(
        SEED
    ),

    "tensorflow_version": (
        tf.__version__
    ),

    "training_images": int(
        len(train_dataframe)
    ),

    "validation_images": int(
        len(validation_dataframe)
    ),

    "batch_size": int(
        BATCH_SIZE
    ),

    "image_height": int(
        IMAGE_HEIGHT
    ),

    "image_width": int(
        IMAGE_WIDTH
    ),

    "negative_class_weight": float(
        class_weights[0]
    ),

    "positive_class_weight": float(
        class_weights[1]
    ),

    "training_batch_minimum": float(
        tf.reduce_min(
            training_images
        ).numpy()
    ),

    "training_batch_maximum": float(
        tf.reduce_max(
            training_images
        ).numpy()
    ),

    "prediction_change_after_training": (
        prediction_change
    ),

    "passed": True,
}


with REPORT_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        report,
        file,
        indent=2,
    )


print("\nGENERATED REPORT")
print("================")

print(
    REPORT_PATH
)


print(
    "\nTRAINING PIPELINE "
    "SMOKE TEST COMPLETED"
)