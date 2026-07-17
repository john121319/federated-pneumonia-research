import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    FIGURES_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MANIFEST_DIR,
    TABLES_DIR,
)

from src.dicom import (
    load_dicom_pixels,
    preprocess_dicom,
)


RANDOM_SEED = 42

SAMPLES_PER_CLASS = 4

TRAIN_MANIFEST_PATH = (
    MANIFEST_DIR
    / "train.csv"
)

OUTPUT_FIGURE_PATH = (
    FIGURES_DIR
    / "rsna_preprocessing_examples.png"
)

OUTPUT_TABLE_PATH = (
    TABLES_DIR
    / "rsna_preprocessing_validation.csv"
)

OUTPUT_REPORT_PATH = (
    TABLES_DIR
    / "rsna_preprocessing_validation.json"
)


print("\nPREPROCESSING VALIDATION")
print("========================")

print(
    "Training manifest:",
    TRAIN_MANIFEST_PATH,
)

print(
    "Manifest exists:",
    TRAIN_MANIFEST_PATH.exists(),
)


if not TRAIN_MANIFEST_PATH.exists():

    raise FileNotFoundError(
        TRAIN_MANIFEST_PATH
    )


FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


train_dataframe = pd.read_csv(
    TRAIN_MANIFEST_PATH
)


required_columns = {
    "exam_id",
    "dicom_path",
    "label",
    "detailed_class",
    "view_position",
}


missing_columns = (
    required_columns
    - set(
        train_dataframe.columns
    )
)


if missing_columns:

    raise ValueError(
        "Training manifest is missing: "
        + str(
            sorted(
                missing_columns
            )
        )
    )


print(
    "Training images:",
    len(train_dataframe),
)


class_order = [
    "Normal",
    "No Lung Opacity / Not Normal",
    "Lung Opacity",
]


selected_groups = []


for class_name in class_order:

    class_dataframe = (
        train_dataframe[
            train_dataframe[
                "detailed_class"
            ]
            == class_name
        ]
        .copy()
    )


    if (
        len(class_dataframe)
        < SAMPLES_PER_CLASS
    ):

        raise ValueError(
            f"Not enough examples for "
            f"class: {class_name}"
        )


    selected_class_dataframe = (
        class_dataframe.sample(
            n=SAMPLES_PER_CLASS,
            random_state=RANDOM_SEED,
        )
    )


    selected_groups.append(
        selected_class_dataframe
    )


selected_dataframe = pd.concat(
    selected_groups,
    ignore_index=True,
)


validation_rows = []

processed_images = []


print("\nSELECTED IMAGE AUDIT")
print("====================")


for row_number, row in (
    selected_dataframe.iterrows()
):

    raw_image = load_dicom_pixels(
        row["dicom_path"]
    )


    processed_image = preprocess_dicom(
        row["dicom_path"]
    )


    processed_images.append(
        processed_image
    )


    finite_values = bool(
        np.isfinite(
            processed_image
        ).all()
    )


    expected_shape = (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )


    correct_shape = (
        processed_image.shape
        == expected_shape
    )


    minimum = float(
        processed_image.min()
    )


    maximum = float(
        processed_image.max()
    )


    standard_deviation = float(
        processed_image.std()
    )


    within_expected_range = (
        minimum >= 0.0
        and maximum <= 1.0
    )


    non_blank = (
        standard_deviation > 0.001
    )


    validation_passed = bool(
        finite_values
        and correct_shape
        and within_expected_range
        and non_blank
    )


    validation_rows.append(
        {
            "exam_id": row["exam_id"],
            "detailed_class": (
                row["detailed_class"]
            ),
            "label": int(
                row["label"]
            ),
            "view_position": (
                row["view_position"]
            ),
            "raw_rows": int(
                raw_image.shape[0]
            ),
            "raw_columns": int(
                raw_image.shape[1]
            ),
            "processed_rows": int(
                processed_image.shape[0]
            ),
            "processed_columns": int(
                processed_image.shape[1]
            ),
            "processed_minimum": minimum,
            "processed_maximum": maximum,
            "processed_mean": float(
                processed_image.mean()
            ),
            "processed_standard_deviation": (
                standard_deviation
            ),
            "finite_values": finite_values,
            "correct_shape": correct_shape,
            "within_expected_range": (
                within_expected_range
            ),
            "non_blank": non_blank,
            "validation_passed": (
                validation_passed
            ),
        }
    )


    print(
        f"{row_number + 1:02d} | "
        f"{row['detailed_class']} | "
        f"{row['view_position']} | "
        f"shape={processed_image.shape} | "
        f"min={minimum:.4f} | "
        f"max={maximum:.4f} | "
        f"std={standard_deviation:.4f} | "
        f"passed={validation_passed}"
    )


validation_dataframe = pd.DataFrame(
    validation_rows
)


validation_dataframe.to_csv(
    OUTPUT_TABLE_PATH,
    index=False,
)


failed_dataframe = (
    validation_dataframe[
        ~validation_dataframe[
            "validation_passed"
        ]
    ]
)


figure, axes = plt.subplots(
    nrows=len(class_order),
    ncols=SAMPLES_PER_CLASS,
    figsize=(14, 10),
)


for image_number, (
    image,
    row,
) in enumerate(
    zip(
        processed_images,
        selected_dataframe.itertuples(
            index=False
        ),
    )
):

    row_index = (
        image_number
        // SAMPLES_PER_CLASS
    )


    column_index = (
        image_number
        % SAMPLES_PER_CLASS
    )


    axis = axes[
        row_index,
        column_index,
    ]


    axis.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )


    axis.set_title(
        f"{row.detailed_class}\n"
        f"{row.view_position} | "
        f"label={row.label}",
        fontsize=9,
    )


    axis.axis("off")


figure.suptitle(
    "RSNA DICOM preprocessing examples",
    fontsize=16,
)


figure.tight_layout(
    rect=[
        0.0,
        0.0,
        1.0,
        0.96,
    ]
)


figure.savefig(
    OUTPUT_FIGURE_PATH,
    dpi=200,
    bbox_inches="tight",
)


plt.close(
    figure
)


validation_report = {
    "sample_count": int(
        len(validation_dataframe)
    ),
    "passed_count": int(
        validation_dataframe[
            "validation_passed"
        ].sum()
    ),
    "failed_count": int(
        len(failed_dataframe)
    ),
    "image_height": (
        IMAGE_HEIGHT
    ),
    "image_width": (
        IMAGE_WIDTH
    ),
    "minimum_processed_value": float(
        validation_dataframe[
            "processed_minimum"
        ].min()
    ),
    "maximum_processed_value": float(
        validation_dataframe[
            "processed_maximum"
        ].max()
    ),
    "minimum_standard_deviation": float(
        validation_dataframe[
            "processed_standard_deviation"
        ].min()
    ),
}


with OUTPUT_REPORT_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        validation_report,
        file,
        indent=2,
    )


print("\nVALIDATION SUMMARY")
print("==================")

print(
    "Images tested:",
    len(validation_dataframe),
)

print(
    "Passed:",
    int(
        validation_dataframe[
            "validation_passed"
        ].sum()
    ),
)

print(
    "Failed:",
    len(failed_dataframe),
)


print("\nGENERATED FILES")
print("===============")

print(
    "Example figure:",
    OUTPUT_FIGURE_PATH,
)

print(
    "Validation table:",
    OUTPUT_TABLE_PATH,
)

print(
    "Validation report:",
    OUTPUT_REPORT_PATH,
)


if len(failed_dataframe) > 0:

    print(
        "\nFAILED EXAMPLES"
    )

    print(
        failed_dataframe.to_string(
            index=False
        )
    )


    raise RuntimeError(
        "Some preprocessing examples failed."
    )


print(
    "\nDICOM PREPROCESSING "
    "VALIDATION COMPLETED"
)