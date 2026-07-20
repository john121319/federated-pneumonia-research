from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


def decode_cached_png(
    image_path,
    label,
):

    image_bytes = tf.io.read_file(
        image_path
    )
    image = tf.io.decode_png(
        image_bytes,
        channels=1,
        dtype=tf.uint16,
    )


    image.set_shape(
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            1,
        ]
    )


    image = tf.cast(
        image,
        tf.float32,
    )


    image = image / 65535.0


    image = tf.clip_by_value(
        image,
        0.0,
        1.0,
    )


    label = tf.cast(
        label,
        tf.float32,
    )


    return image, label


def create_dataset_from_dataframe(
    dataframe,
    training,
    seed,
    batch_size=BATCH_SIZE,
):

    required_columns = {
        "cache_path",
        "label",
    }


    missing_columns = (
        required_columns
        - set(
            dataframe.columns
        )
    )


    if missing_columns:

        raise ValueError(
            "Manifest is missing columns: "
            + str(
                sorted(
                    missing_columns
                )
            )
        )


    if len(dataframe) == 0:

        raise ValueError(
            "Cannot create a dataset "
            "from an empty DataFrame."
        )


    image_paths = (
        dataframe[
            "cache_path"
        ]
        .astype(str)
        .to_numpy()
    )


    labels = (
        dataframe[
            "label"
        ]
        .astype(np.float32)
        .to_numpy()
    )


    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                image_paths,
                labels,
            )
        )
    )


    if training:

        dataset = dataset.shuffle(
            buffer_size=min(
                len(dataframe),
                10000,
            ),
            seed=seed,
            reshuffle_each_iteration=True,
        )


    dataset = dataset.map(
        decode_cached_png,
        num_parallel_calls=(
            tf.data.AUTOTUNE
        ),
        deterministic=(
            not training
        ),
    )


    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,
    )


    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )


    return dataset


def create_dataset(
    manifest_path,
    training,
    seed,
    batch_size=BATCH_SIZE,
):

    manifest_path = Path(
        manifest_path
    )


    if not manifest_path.exists():

        raise FileNotFoundError(
            manifest_path
        )


    dataframe = pd.read_csv(
        manifest_path
    )


    return create_dataset_from_dataframe(
        dataframe=dataframe,
        training=training,
        seed=seed,
        batch_size=batch_size,
    )


def calculate_class_weights(
    dataframe,
):

    positive_count = int(
        (
            dataframe[
                "label"
            ]
            == 1
        ).sum()
    )


    negative_count = int(
        (
            dataframe[
                "label"
            ]
            == 0
        ).sum()
    )


    total_count = (
        positive_count
        + negative_count
    )


    if positive_count == 0:

        raise ValueError(
            "No positive samples were found."
        )


    if negative_count == 0:

        raise ValueError(
            "No negative samples were found."
        )


    negative_weight = (
        total_count
        / (
            2.0
            * negative_count
        )
    )


    positive_weight = (
        total_count
        / (
            2.0
            * positive_count
        )
    )


    return {
        0: float(
            negative_weight
        ),

        1: float(
            positive_weight
        ),
    }