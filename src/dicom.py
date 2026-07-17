from pathlib import Path

import numpy as np
import pydicom
import tensorflow as tf

from config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


def load_dicom_pixels(
    dicom_path,
):

    dicom_path = Path(
        dicom_path
    )


    dicom = pydicom.dcmread(
        dicom_path
    )


    image = dicom.pixel_array.astype(
        np.float32
    )

    rescale_slope = float(
        getattr(
            dicom,
            "RescaleSlope",
            1.0,
        )
    )


    rescale_intercept = float(
        getattr(
            dicom,
            "RescaleIntercept",
            0.0,
        )
    )


    image = (
        image * rescale_slope
        + rescale_intercept
    )


    photometric = str(
        getattr(
            dicom,
            "PhotometricInterpretation",
            "",
        )
    ).strip()

    if photometric == "MONOCHROME1":

        image = (
            image.max()
            + image.min()
            - image
        )


    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


    return image


def normalize_xray(
    image,
    lower_percentile=0.5,
    upper_percentile=99.5,
):

    image = np.asarray(
        image,
        dtype=np.float32,
    )


    lower_value = np.percentile(
        image,
        lower_percentile,
    )


    upper_value = np.percentile(
        image,
        upper_percentile,
    )


    if upper_value <= lower_value:

        minimum = float(
            image.min()
        )

        maximum = float(
            image.max()
        )


        if maximum <= minimum:

            return np.zeros_like(
                image,
                dtype=np.float32,
            )


        normalized = (
            image - minimum
        ) / (
            maximum - minimum
        )


        return normalized.astype(
            np.float32
        )


    image = np.clip(
        image,
        lower_value,
        upper_value,
    )


    image = (
        image - lower_value
    ) / (
        upper_value - lower_value
    )


    return image.astype(
        np.float32
    )


def resize_xray(
    image,
    height=IMAGE_HEIGHT,
    width=IMAGE_WIDTH,
):

    image_tensor = tf.convert_to_tensor(
        image,
        dtype=tf.float32,
    )


    image_tensor = tf.expand_dims(
        image_tensor,
        axis=-1,
    )


    resized_tensor = tf.image.resize(
        image_tensor,
        size=[
            height,
            width,
        ],
        method="bilinear",
        antialias=True,
    )


    resized_image = (
        resized_tensor
        .numpy()
        .squeeze(axis=-1)
    )


    return resized_image.astype(
        np.float32
    )


def preprocess_dicom(
    dicom_path,
    height=IMAGE_HEIGHT,
    width=IMAGE_WIDTH,
):

    image = load_dicom_pixels(
        dicom_path
    )


    image = normalize_xray(
        image
    )


    image = resize_xray(
        image=image,
        height=height,
        width=width,
    )


    image = np.clip(
        image,
        0.0,
        1.0,
    )


    return image.astype(
        np.float32
    )