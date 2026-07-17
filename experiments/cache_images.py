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

from src.dicom import preprocess_dicom

OVERWRITE_EXISTING = False

FULL_MANIFEST_PATH = (
    MANIFEST_DIR
    / "full_manifest.csv"
)

CACHE_IMAGE_DIR = (
    CACHE_DIR
    / "images"
)

CACHED_FULL_MANIFEST_PATH = (
    MANIFEST_DIR
    / "full_manifest_cached.csv"
)

CACHED_TRAIN_MANIFEST_PATH = (
    MANIFEST_DIR
    / "train_cached.csv"
)

CACHED_VALIDATION_MANIFEST_PATH = (
    MANIFEST_DIR
    / "validation_cached.csv"
)

CACHED_TEST_MANIFEST_PATH = (
    MANIFEST_DIR
    / "test_cached.csv"
)

CACHE_ERROR_PATH = (
    TABLES_DIR
    / "rsna_cache_errors.csv"
)

CACHE_REPORT_PATH = (
    TABLES_DIR
    / "rsna_cache_report.json"
)

CACHE_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("\nRSNA IMAGE CACHE")
print("================")

print(
    "Source manifest:",
    FULL_MANIFEST_PATH,
)

print(
    "Source manifest exists:",
    FULL_MANIFEST_PATH.exists(),
)

print(
    "Cache directory:",
    CACHE_IMAGE_DIR,
)


if not FULL_MANIFEST_PATH.exists():

    raise FileNotFoundError(
        FULL_MANIFEST_PATH
    )

full_manifest = pd.read_csv(
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
    - set(full_manifest.columns)
)


if missing_columns:

    raise ValueError(
        "Manifest is missing columns: "
        + str(sorted(missing_columns))
    )


if full_manifest["exam_id"].duplicated().any():

    raise ValueError(
        "Duplicate examination IDs were found."
    )


print(
    "Images to process:",
    len(full_manifest),
)

print(
    "Target dimensions:",
    f"{IMAGE_HEIGHT} × {IMAGE_WIDTH}",
)

cache_paths = []

cache_statuses = []

cache_minimums = []

cache_maximums = []

cache_means = []

cache_standard_deviations = []

error_rows = []

created_count = 0

existing_count = 0

start_time = time.time()


for number, row in enumerate(
    full_manifest.itertuples(index=False),
    start=1,
):

    exam_id = str(row.exam_id)

    dicom_path = Path(
        row.dicom_path
    )

    cache_path = (
        CACHE_IMAGE_DIR
        / f"{exam_id}.png"
    )


    try:

        if (
            cache_path.exists()
            and not OVERWRITE_EXISTING
        ):

            existing_count += 1

            cached_image = np.asarray(
                Image.open(cache_path),
                dtype=np.uint16,
            )

            processed_image = (
                cached_image.astype(np.float32)
                / 65535.0
            )

            cache_status = "existing"


        else:

            processed_image = preprocess_dicom(
                dicom_path
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


            processed_image = np.clip(
                processed_image,
                0.0,
                1.0,
            )


            if float(
                processed_image.std()
            ) <= 0.001:

                raise ValueError(
                    "Processed image appears blank."
                )


            image_uint16 = np.round(
                processed_image * 65535.0
            ).astype(np.uint16)


            Image.fromarray(
                image_uint16
            ).save(
                cache_path,
                format="PNG",
                optimize=True,
            )


            created_count += 1

            cache_status = "created"


        if processed_image.shape != (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ):

            raise ValueError(
                "Cached image has unexpected shape: "
                f"{processed_image.shape}"
            )


        minimum = float(
            processed_image.min()
        )

        maximum = float(
            processed_image.max()
        )

        mean = float(
            processed_image.mean()
        )

        standard_deviation = float(
            processed_image.std()
        )


        if (
            minimum < 0.0
            or maximum > 1.0
        ):

            raise ValueError(
                "Cached image is outside [0,1]."
            )


        if standard_deviation <= 0.001:

            raise ValueError(
                "Cached image appears blank."
            )


        cache_paths.append(
            str(cache_path.resolve())
        )

        cache_statuses.append(
            cache_status
        )

        cache_minimums.append(
            minimum
        )

        cache_maximums.append(
            maximum
        )

        cache_means.append(
            mean
        )

        cache_standard_deviations.append(
            standard_deviation
        )


    except Exception as error:

        cache_paths.append(None)

        cache_statuses.append("error")

        cache_minimums.append(None)

        cache_maximums.append(None)

        cache_means.append(None)

        cache_standard_deviations.append(None)


        error_rows.append(
            {
                "exam_id": exam_id,
                "dicom_path": str(
                    dicom_path
                ),
                "cache_path": str(
                    cache_path
                ),
                "error": repr(error),
            }
        )


    if number % 500 == 0:

        elapsed_seconds = (
            time.time()
            - start_time
        )

        processing_rate = (
            number
            / elapsed_seconds
        )

        remaining_images = (
            len(full_manifest)
            - number
        )

        estimated_remaining_seconds = (
            remaining_images
            / processing_rate
            if processing_rate > 0
            else 0
        )


        print(
            f"Processed {number:,}/"
            f"{len(full_manifest):,} | "
            f"created={created_count:,} | "
            f"existing={existing_count:,} | "
            f"errors={len(error_rows):,} | "
            f"rate={processing_rate:.2f}/s | "
            f"remaining≈"
            f"{estimated_remaining_seconds / 60:.1f} min"
        )

full_manifest["cache_path"] = (
    cache_paths
)

full_manifest["cache_status"] = (
    cache_statuses
)

full_manifest["cache_minimum"] = (
    cache_minimums
)

full_manifest["cache_maximum"] = (
    cache_maximums
)

full_manifest["cache_mean"] = (
    cache_means
)

full_manifest[
    "cache_standard_deviation"
] = cache_standard_deviations


successful_manifest = (
    full_manifest[
        full_manifest[
            "cache_status"
        ]
        != "error"
    ]
    .copy()
)

if error_rows:

    pd.DataFrame(
        error_rows
    ).to_csv(
        CACHE_ERROR_PATH,
        index=False,
    )

if error_rows:

    print("\nCACHE ERRORS")
    print("============")

    print(
        "Failed images:",
        len(error_rows),
    )

    print(
        "Error report:",
        CACHE_ERROR_PATH,
    )

    raise RuntimeError(
        "Cache generation did not complete "
        "because some images failed."
    )

successful_manifest.to_csv(
    CACHED_FULL_MANIFEST_PATH,
    index=False,
)


train_manifest = successful_manifest[
    successful_manifest["split"]
    == "train"
].copy()


validation_manifest = successful_manifest[
    successful_manifest["split"]
    == "validation"
].copy()


test_manifest = successful_manifest[
    successful_manifest["split"]
    == "test"
].copy()


train_manifest.to_csv(
    CACHED_TRAIN_MANIFEST_PATH,
    index=False,
)


validation_manifest.to_csv(
    CACHED_VALIDATION_MANIFEST_PATH,
    index=False,
)


test_manifest.to_csv(
    CACHED_TEST_MANIFEST_PATH,
    index=False,
)

cache_files = list(
    CACHE_IMAGE_DIR.glob("*.png")
)


total_cache_bytes = sum(
    path.stat().st_size
    for path in cache_files
)


elapsed_seconds = (
    time.time()
    - start_time
)

cache_report = {
    "image_count": int(
        len(full_manifest)
    ),
    "created_count": int(
        created_count
    ),
    "existing_count": int(
        existing_count
    ),
    "error_count": int(
        len(error_rows)
    ),
    "cached_file_count": int(
        len(cache_files)
    ),
    "image_height": int(
        IMAGE_HEIGHT
    ),
    "image_width": int(
        IMAGE_WIDTH
    ),
    "cache_format": (
        "16-bit grayscale PNG"
    ),
    "cache_size_bytes": int(
        total_cache_bytes
    ),
    "cache_size_megabytes": float(
        total_cache_bytes
        / (1024 ** 2)
    ),
    "elapsed_seconds": float(
        elapsed_seconds
    ),
    "minimum_intensity": float(
        successful_manifest[
            "cache_minimum"
        ].min()
    ),
    "maximum_intensity": float(
        successful_manifest[
            "cache_maximum"
        ].max()
    ),
    "minimum_standard_deviation": float(
        successful_manifest[
            "cache_standard_deviation"
        ].min()
    ),
    "maximum_standard_deviation": float(
        successful_manifest[
            "cache_standard_deviation"
        ].max()
    ),
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

print("\nCACHE SUMMARY")
print("=============")

print(
    "Manifest images:",
    len(full_manifest),
)

print(
    "Created images:",
    created_count,
)

print(
    "Previously existing images:",
    existing_count,
)

print(
    "Errors:",
    len(error_rows),
)

print(
    "PNG files in cache:",
    len(cache_files),
)

print(
    "Training cache records:",
    len(train_manifest),
)

print(
    "Validation cache records:",
    len(validation_manifest),
)

print(
    "Test cache records:",
    len(test_manifest),
)

print(
    "Cache size:",
    f"{total_cache_bytes / (1024 ** 2):.2f} MB",
)

print(
    "Elapsed time:",
    f"{elapsed_seconds / 60:.2f} minutes",
)

print(
    "Minimum image standard deviation:",
    f"{successful_manifest['cache_standard_deviation'].min():.6f}",
)


print("\nGENERATED MANIFESTS")
print("===================")

print(
    CACHED_FULL_MANIFEST_PATH
)

print(
    CACHED_TRAIN_MANIFEST_PATH
)

print(
    CACHED_VALIDATION_MANIFEST_PATH
)

print(
    CACHED_TEST_MANIFEST_PATH
)

print(
    CACHE_REPORT_PATH
)


print(
    "\nRSNA IMAGE CACHE COMPLETED"
)