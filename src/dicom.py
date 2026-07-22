from pathlib import Path

import numpy as np
import pydicom
import tensorflow as tf

from config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

LOWER_PERCENTILE = 0.5
UPPER_PERCENTILE = 99.5

def load_dicom_pixels(
    dicom_path,
):

    dicom_path = Path(
        dicom_path
    )


    if not dicom_path.exists():

        raise FileNotFoundError(
            dicom_path
        )


    dicom_dataset = pydicom.dcmread(
        str(
            dicom_path
        )
    )


    pixel_array = np.asarray(
        dicom_dataset.pixel_array,
        dtype=np.float32,
    )


    if pixel_array.ndim != 2:

        raise ValueError(
            "Expected a two-dimensional grayscale "
            f"DICOM image, but received shape "
            f"{pixel_array.shape}."
        )


    return pixel_array

def preprocess_dicom_with_metadata(
    dicom_path,
    target_height=IMAGE_HEIGHT,
    target_width=IMAGE_WIDTH,
):

    dicom_path = Path(
        dicom_path
    )


    if not dicom_path.exists():

        raise FileNotFoundError(
            dicom_path
        )

    dicom_dataset = pydicom.dcmread(
        str(
            dicom_path
        )
    )


    pixel_array = np.asarray(
        dicom_dataset.pixel_array,
        dtype=np.float32,
    )


    if pixel_array.ndim != 2:

        raise ValueError(
            "Expected a two-dimensional grayscale "
            f"DICOM image, but received shape "
            f"{pixel_array.shape}."
        )


    source_height = int(
        pixel_array.shape[0]
    )


    source_width = int(
        pixel_array.shape[1]
    )

    rescale_slope = float(
        getattr(
            dicom_dataset,
            "RescaleSlope",
            1.0,
        )
    )


    rescale_intercept = float(
        getattr(
            dicom_dataset,
            "RescaleIntercept",
            0.0,
        )
    )
    image = (
        pixel_array
        * rescale_slope
        + rescale_intercept
    )

    finite_values = image[
        np.isfinite(
            image
        )
    ]


    if finite_values.size == 0:

        raise ValueError(
            "The DICOM image contains no "
            "finite pixel values."
        )


    finite_minimum = float(
        finite_values.min()
    )


    finite_maximum = float(
        finite_values.max()
    )


    image = np.nan_to_num(
        image,
        nan=finite_minimum,
        posinf=finite_maximum,
        neginf=finite_minimum,
    )

    photometric_interpretation = str(
        getattr(
            dicom_dataset,
            "PhotometricInterpretation",
            "",
        )
    ).upper()


    was_inverted = False


    if (
        photometric_interpretation
        == "MONOCHROME1"
    ):

        image = (
            image.max()
            + image.min()
            - image
        )


        was_inverted = True

    lower_value = float(
        np.percentile(
            image,
            LOWER_PERCENTILE,
        )
    )


    upper_value = float(
        np.percentile(
            image,
            UPPER_PERCENTILE,
        )
    )


    if not np.isfinite(
        lower_value
    ):

        lower_value = float(
            image.min()
        )


    if not np.isfinite(
        upper_value
    ):

        upper_value = float(
            image.max()
        )


    if upper_value <= lower_value:

        lower_value = float(
            image.min()
        )


        upper_value = float(
            image.max()
        )


    if upper_value <= lower_value:

        raise ValueError(
            "The DICOM image has no usable "
            "intensity variation."
        )

    image = np.clip(
        image,
        lower_value,
        upper_value,
    )

    image = (
        image
        - lower_value
    ) / (
        upper_value
        - lower_value
    )


    image = np.clip(
        image,
        0.0,
        1.0,
    )

    image_tensor = tf.convert_to_tensor(
        image[
            ...,
            np.newaxis
        ],
        dtype=tf.float32,
    )


    resized_tensor = tf.image.resize(
        image_tensor,
        size=[
            target_height,
            target_width,
        ],
        method="bilinear",
        antialias=True,
    )


    resized_image = (
        resized_tensor
        .numpy()
        .squeeze(
            axis=-1
        )
    )


    resized_image = np.clip(
        resized_image,
        0.0,
        1.0,
    ).astype(
        np.float32
    )

    metadata = {
        "dicom_path": str(
            dicom_path
        ),

        "source_height": (
            source_height
        ),

        "source_width": (
            source_width
        ),

        "target_height": int(
            target_height
        ),

        "target_width": int(
            target_width
        ),

        "photometric_interpretation": (
            photometric_interpretation
        ),

        "was_inverted": bool(
            was_inverted
        ),

        "rescale_slope": float(
            rescale_slope
        ),

        "rescale_intercept": float(
            rescale_intercept
        ),

        "lower_percentile": float(
            LOWER_PERCENTILE
        ),

        "upper_percentile": float(
            UPPER_PERCENTILE
        ),

        "lower_clipping_value": float(
            lower_value
        ),

        "upper_clipping_value": float(
            upper_value
        ),
    }


    return (
        resized_image,
        metadata,
    )

def preprocess_dicom(
    dicom_path,
    target_height=IMAGE_HEIGHT,
    target_width=IMAGE_WIDTH,
):

    image, _ = (
        preprocess_dicom_with_metadata(
            dicom_path=(
                dicom_path
            ),

            target_height=(
                target_height
            ),

            target_width=(
                target_width
            ),
        )
    )


    return image