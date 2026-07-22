import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import (
    CACHE_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MANIFEST_DIR,
    TABLES_DIR,
)

from src.dicom import (
    preprocess_dicom,
)

OVERWRITE_EXISTING_IMAGES = False

PROGRESS_INTERVAL = 1000

BLANK_IMAGE_STD_THRESHOLD = 0.001

FULL_MANIFEST_PATH = (
    MANIFEST_DIR
    / "full_manifest.csv"
)


SOURCE_SPLIT_MANIFESTS = {
    "train": (
        MANIFEST_DIR
        / "train.csv"
    ),

    "validation": (
        MANIFEST_DIR
        / "validation.csv"
    ),

    "test": (
        MANIFEST_DIR
        / "test.csv"
    ),
}

CACHE_IMAGE_DIR = (
    CACHE_DIR
    / "images"
)


CACHED_MANIFEST_PATHS = {
    "full": (
        MANIFEST_DIR
        / "full_manifest_cached.csv"
    ),

    "train": (
        MANIFEST_DIR
        / "train_cached.csv"
    ),

    "validation": (
        MANIFEST_DIR
        / "validation_cached.csv"
    ),

    "test": (
        MANIFEST_DIR
        / "test_cached.csv"
    ),
}


CACHE_REPORT_PATH = (
    TABLES_DIR
    / "rsna_cache_report.json"
)


CACHE_ERROR_PATH = (
    TABLES_DIR
    / "rsna_cache_errors.csv"
)

CACHE_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    "\nRSNA IMAGE CACHE"
)


print(
    "================"
)


print(
    "Source manifest:",
    FULL_MANIFEST_PATH,
)


print(
    "Manifest exists:",
    FULL_MANIFEST_PATH.exists(),
)


print(
    "Cache directory:",
    CACHE_IMAGE_DIR,
)


print(
    "Overwrite existing images:",
    OVERWRITE_EXISTING_IMAGES,
)


if not FULL_MANIFEST_PATH.exists():

    raise FileNotFoundError(
        FULL_MANIFEST_PATH
    )


full_dataframe = pd.read_csv(
    FULL_MANIFEST_PATH,
    dtype={
        "exam_id": str,

        "original_patient_id": str,
    },
)


required_columns = {
    "exam_id",
    "dicom_path",
    "label",
    "split",
}


missing_columns = (
    required_columns
    - set(
        full_dataframe.columns
    )
)


if missing_columns:

    raise ValueError(
        "Full manifest is missing columns: "
        + str(
            sorted(
                missing_columns
            )
        )
    )


if full_dataframe[
    "exam_id"
].duplicated().any():

    raise ValueError(
        "The full manifest contains duplicate "
        "examination identifiers."
    )


print(
    "Images to process:",
    len(
        full_dataframe
    ),
)


print(
    "Target dimensions:",
    f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}",
)
start_time = time.time()


created_count = 0

reused_count = 0

error_rows = []

cache_path_by_exam = {}


minimum_standard_deviation = float(
    "inf"
)


maximum_standard_deviation = 0.0


for image_number, row in enumerate(
    full_dataframe.itertuples(
        index=False
    ),
    start=1,
):

    exam_id = str(
        row.exam_id
    )


    dicom_path = Path(
        str(
            row.dicom_path
        )
    )


    output_path = (
        CACHE_IMAGE_DIR
        / f"{exam_id}.png"
    )


    cache_path_by_exam[
        exam_id
    ] = str(
        output_path.resolve()
    )


    try:

        if (
            output_path.exists()
            and not OVERWRITE_EXISTING_IMAGES
        ):

            reused_count += 1


        else:

            processed_image = (
                preprocess_dicom(
                    dicom_path
                )
            )


            if processed_image.shape != (
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
            ):

                raise ValueError(
                    "Unexpected processed shape: "
                    f"{processed_image.shape}"
                )


            if not np.isfinite(
                processed_image
            ).all():

                raise ValueError(
                    "Processed image contains "
                    "non-finite values."
                )


            minimum_value = float(
                processed_image.min()
            )


            maximum_value = float(
                processed_image.max()
            )


            standard_deviation = float(
                processed_image.std()
            )


            if (
                minimum_value < 0.0
                or maximum_value > 1.0
            ):

                raise ValueError(
                    "Processed values are outside "
                    "the range [0, 1]."
                )


            if (
                standard_deviation
                <= BLANK_IMAGE_STD_THRESHOLD
            ):

                raise ValueError(
                    "Processed image may be blank. "
                    f"Standard deviation="
                    f"{standard_deviation}"
                )


            minimum_standard_deviation = min(
                minimum_standard_deviation,
                standard_deviation,
            )


            maximum_standard_deviation = max(
                maximum_standard_deviation,
                standard_deviation,
            )
            image_uint16 = np.rint(
                np.clip(
                    processed_image,
                    0.0,
                    1.0,
                )
                * 65535.0
            ).astype(
                np.uint16
            )
            Image.fromarray(
                image_uint16
            ).save(
                output_path,
                format="PNG",
                compress_level=4,
            )


            created_count += 1


    except Exception as error:

        error_rows.append(
            {
                "exam_id": exam_id,

                "dicom_path": str(
                    dicom_path
                ),

                "cache_path": str(
                    output_path
                ),

                "error": repr(
                    error
                ),
            }
        )


    if (
        image_number
        % PROGRESS_INTERVAL
        == 0
        or image_number
        == len(
            full_dataframe
        )
    ):

        elapsed_seconds = (
            time.time()
            - start_time
        )


        processing_rate = (
            image_number
            / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        )


        print(
            f"Processed "
            f"{image_number:,}/"
            f"{len(full_dataframe):,} | "
            f"created={created_count:,} | "
            f"reused={reused_count:,} | "
            f"errors={len(error_rows):,} | "
            f"rate={processing_rate:.2f} images/s"
        )


elapsed_seconds = (
    time.time()
    - start_time
)
error_dataframe = pd.DataFrame(
    error_rows,
    columns=[
        "exam_id",
        "dicom_path",
        "cache_path",
        "error",
    ],
)


error_dataframe.to_csv(
    CACHE_ERROR_PATH,
    index=False,
)
cached_png_paths = list(
    CACHE_IMAGE_DIR.glob(
        "*.png"
    )
)


cached_file_count = len(
    cached_png_paths
)


expected_image_count = len(
    full_dataframe
)


print(
    "\nCACHE FILE VERIFICATION"
)


print(
    "======================="
)


print(
    "Expected images:",
    expected_image_count,
)


print(
    "PNG files present:",
    cached_file_count,
)


print(
    "Created:",
    created_count,
)


print(
    "Reused:",
    reused_count,
)


print(
    "Errors:",
    len(
        error_rows
    ),
)


if len(error_rows) > 0:

    raise RuntimeError(
        "One or more DICOM files could not "
        "be cached. Inspect: "
        f"{CACHE_ERROR_PATH}"
    )


missing_cached_paths = []


for exam_id in (
    full_dataframe[
        "exam_id"
    ].astype(str)
):

    expected_path = (
        CACHE_IMAGE_DIR
        / f"{exam_id}.png"
    )


    if not expected_path.exists():

        missing_cached_paths.append(
            str(
                expected_path
            )
        )


if missing_cached_paths:

    raise RuntimeError(
        "Some expected cached images are missing. "
        f"Missing count: "
        f"{len(missing_cached_paths)}"
    )
sample_count = min(
    12,
    len(
        full_dataframe
    ),
)


sample_dataframe = full_dataframe.sample(
    n=sample_count,
    random_state=42,
)


saved_png_modes = set()


for row in sample_dataframe.itertuples(
    index=False
):

    exam_id = str(
        row.exam_id
    )


    cached_path = (
        CACHE_IMAGE_DIR
        / f"{exam_id}.png"
    )


    with Image.open(
        cached_path
    ) as saved_image:

        saved_png_modes.add(
            saved_image.mode
        )


        saved_array = np.asarray(
            saved_image
        )


    if saved_array.shape != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ):

        raise RuntimeError(
            "Saved PNG has incorrect shape: "
            f"{cached_path} "
            f"{saved_array.shape}"
        )


    if not np.isfinite(
        saved_array
    ).all():

        raise RuntimeError(
            "Saved PNG contains non-finite "
            f"values: {cached_path}"
        )


print(
    "Representative PNG files checked:",
    sample_count,
)


print(
    "Observed Pillow image modes:",
    sorted(
        saved_png_modes
    ),
)
print(
    "\nGENERATING CACHED MANIFESTS"
)


print(
    "==========================="
)


all_source_manifests = {
    "full": (
        FULL_MANIFEST_PATH
    ),

    **SOURCE_SPLIT_MANIFESTS,
}


split_counts = {}


for manifest_name, source_path in (
    all_source_manifests.items()
):

    if not source_path.exists():

        raise FileNotFoundError(
            source_path
        )


    manifest_dataframe = pd.read_csv(
        source_path,
        dtype={
            "exam_id": str,

            "original_patient_id": str,
        },
    )


    manifest_dataframe[
        "cache_path"
    ] = (
        manifest_dataframe[
            "exam_id"
        ]
        .astype(str)
        .map(
            cache_path_by_exam
        )
    )


    missing_cache_values = (
        manifest_dataframe[
            "cache_path"
        ].isna()
    )


    if missing_cache_values.any():

        raise RuntimeError(
            f"{manifest_name} manifest contains "
            "examinations without cache paths."
        )


    missing_files = [
        cache_path
        for cache_path
        in manifest_dataframe[
            "cache_path"
        ]
        if not Path(
            cache_path
        ).exists()
    ]


    if missing_files:

        raise RuntimeError(
            f"{manifest_name} manifest contains "
            f"{len(missing_files)} missing "
            "cached files."
        )


    output_path = (
        CACHED_MANIFEST_PATHS[
            manifest_name
        ]
    )


    manifest_dataframe.to_csv(
        output_path,
        index=False,
    )


    split_counts[
        manifest_name
    ] = int(
        len(
            manifest_dataframe
        )
    )


    print(
        f"{manifest_name}: "
        f"{len(manifest_dataframe):,} images "
        f"-> {output_path}"
    )
cache_size_bytes = sum(
    path.stat().st_size
    for path
    in cached_png_paths
)


cache_size_megabytes = (
    cache_size_bytes
    / (
        1024.0
        * 1024.0
    )
)


if minimum_standard_deviation == float(
    "inf"
):

    minimum_standard_deviation = None
cache_report = {
    "source_manifest": str(
        FULL_MANIFEST_PATH
    ),

    "cache_directory": str(
        CACHE_IMAGE_DIR
    ),

    "expected_images": int(
        expected_image_count
    ),

    "cached_png_files": int(
        cached_file_count
    ),

    "created_images": int(
        created_count
    ),

    "reused_images": int(
        reused_count
    ),

    "error_count": int(
        len(
            error_rows
        )
    ),

    "target_height": int(
        IMAGE_HEIGHT
    ),

    "target_width": int(
        IMAGE_WIDTH
    ),

    "storage_format": (
        "lossless 16-bit grayscale PNG"
    ),

    "normalization_range": [
        0.0,
        1.0,
    ],

    "blank_image_standard_deviation_threshold": float(
        BLANK_IMAGE_STD_THRESHOLD
    ),

    "minimum_processed_standard_deviation": (
        minimum_standard_deviation
    ),

    "maximum_processed_standard_deviation": float(
        maximum_standard_deviation
    ),

    "cache_size_bytes": int(
        cache_size_bytes
    ),

    "cache_size_megabytes": float(
        cache_size_megabytes
    ),

    "elapsed_seconds": float(
        elapsed_seconds
    ),

    "elapsed_minutes": float(
        elapsed_seconds
        / 60.0
    ),

    "average_images_per_second": float(
        expected_image_count
        / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    ),

    "representative_png_files_checked": int(
        sample_count
    ),

    "observed_pillow_modes": sorted(
        saved_png_modes
    ),

    "cached_manifest_counts": (
        split_counts
    ),

    "completed_successfully": True,
}


with CACHE_REPORT_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        cache_report,
        file,
        indent=2,
    )

print(
    "\nCACHE SUMMARY"
)


print(
    "============="
)


print(
    "Cached images:",
    cached_file_count,
)


print(
    "Cache size:",
    f"{cache_size_megabytes:.2f} MB",
)


print(
    "Elapsed time:",
    f"{elapsed_seconds / 60.0:.2f} minutes",
)


print(
    "Average rate:",
    f"{expected_image_count / elapsed_seconds:.2f} "
    "images/s",
)


print(
    "Minimum processed standard deviation:",
    minimum_standard_deviation,
)


print(
    "Maximum processed standard deviation:",
    maximum_standard_deviation,
)


print(
    "\nGENERATED FILES"
)


print(
    "==============="
)


for manifest_name, output_path in (
    CACHED_MANIFEST_PATHS.items()
):

    print(
        f"{manifest_name} cached manifest:",
        output_path,
    )


print(
    "Cache report:",
    CACHE_REPORT_PATH,
)


print(
    "Error report:",
    CACHE_ERROR_PATH,
)


print(
    "\nRSNA IMAGE CACHE COMPLETED"
)