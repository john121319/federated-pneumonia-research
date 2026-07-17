import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pydicom

from config import (
    MANIFEST_DIR,
    RSNA_CLASS_INFO_CSV,
    RSNA_IMAGE_DIR,
    RSNA_LABELS_CSV,
    RSNA_MAPPING_JSON,
    TABLES_DIR,
)


def parse_patient_age(value):

    if value is None:

        return None


    text = str(value).strip()


    if not text:

        return None


    try:

        if text.endswith("Y"):

            return float(text[:-1])


        if text.endswith("M"):

            return float(text[:-1]) / 12.0


        if text.endswith("W"):

            return float(text[:-1]) / 52.1429


        if text.endswith("D"):

            return float(text[:-1]) / 365.25


        return float(text)


    except ValueError:

        return None


def get_dicom_value(
    dicom,
    attribute_name,
):

    value = getattr(
        dicom,
        attribute_name,
        None,
    )


    if value is None:

        return None


    return str(value).strip()


def count_json_keys(
    value,
    key_counter,
):

    if isinstance(value, dict):

        for key, nested_value in value.items():

            key_counter[str(key)] += 1

            count_json_keys(
                nested_value,
                key_counter,
            )


    elif isinstance(value, list):

        for item in value:

            count_json_keys(
                item,
                key_counter,
            )


MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


required_paths = [
    RSNA_IMAGE_DIR,
    RSNA_LABELS_CSV,
    RSNA_CLASS_INFO_CSV,
    RSNA_MAPPING_JSON,
]


print("\nREQUIRED FILE CHECK")

print("===================")


for path in required_paths:

    status = (
        "FOUND"
        if path.exists()
        else "MISSING"
    )

    print(
        f"{status}: {path}"
    )


missing_paths = [
    path
    for path in required_paths
    if not path.exists()
]


if missing_paths:

    raise FileNotFoundError(

        "The following required files are missing:\n"

        + "\n".join(
            str(path)
            for path in missing_paths
        )

    )


labels_dataframe = pd.read_csv(
    RSNA_LABELS_CSV
)


required_label_columns = {
    "patientId",
    "x",
    "y",
    "width",
    "height",
    "Target",
}


missing_label_columns = (

    required_label_columns

    - set(
        labels_dataframe.columns
    )

)


if missing_label_columns:

    raise ValueError(

        "Missing columns from stage_2_train_labels.csv: "

        + str(
            sorted(
                missing_label_columns
            )
        )

    )


print("\nRAW LABEL CSV")

print("=============")

print(
    "Rows:",
    len(labels_dataframe),
)

print(
    "Columns:",
    labels_dataframe.columns.tolist(),
)

print(
    "Unique examination IDs:",
    labels_dataframe[
        "patientId"
    ].nunique(),
)

print(
    "\nRaw Target counts:"
)

print(
    labels_dataframe[
        "Target"
    ].value_counts(
        dropna=False
    )
)

exam_labels = (

    labels_dataframe

    .groupby(
        "patientId",
        as_index=False,
    )

    .agg(

        label=(
            "Target",
            "max",
        ),

        annotation_rows=(
            "Target",
            "size",
        ),

        box_count=(
            "Target",
            "sum",
        ),

        x_min=(
            "x",
            "min",
        ),

        y_min=(
            "y",
            "min",
        ),

        x_maximum_start=(
            "x",
            "max",
        ),

        y_maximum_start=(
            "y",
            "max",
        ),

    )

)


exam_labels["label"] = (

    exam_labels["label"]

    .astype(int)

)


exam_labels["box_count"] = (

    exam_labels["box_count"]

    .astype(int)

)


print("\nIMAGE-LEVEL LABELS")

print("==================")

print(
    "Unique examinations:",
    len(exam_labels),
)

print(
    "Negative examinations:",
    int(
        (
            exam_labels["label"] == 0
        ).sum()
    ),
)

print(
    "Positive examinations:",
    int(
        (
            exam_labels["label"] == 1
        ).sum()
    ),
)

print(
    "Examinations with multiple bounding boxes:",
    int(
        (
            exam_labels["box_count"] > 1
        ).sum()
    ),
)

print(
    "Largest number of boxes in one examination:",
    int(
        exam_labels["box_count"].max()
    ),
)


class_dataframe = pd.read_csv(
    RSNA_CLASS_INFO_CSV
)


required_class_columns = {
    "patientId",
    "class",
}


missing_class_columns = (

    required_class_columns

    - set(
        class_dataframe.columns
    )

)


if missing_class_columns:

    raise ValueError(

        "Missing columns from "
        "stage_2_detailed_class_info.csv: "

        + str(
            sorted(
                missing_class_columns
            )
        )

    )


print("\nDETAILED CLASS CSV")

print("==================")

print(
    "Rows:",
    len(class_dataframe),
)

print(
    "Unique examination IDs:",
    class_dataframe[
        "patientId"
    ].nunique(),
)

print(
    "\nDetailed-class row counts:"
)

print(
    class_dataframe[
        "class"
    ].value_counts(
        dropna=False
    )
)


class_conflicts = (

    class_dataframe

    .groupby("patientId")["class"]

    .nunique()

)


conflicting_class_ids = (

    class_conflicts[
        class_conflicts > 1
    ]

    .index

    .tolist()

)


print(
    "\nExaminations with conflicting detailed classes:",
    len(conflicting_class_ids),
)


if conflicting_class_ids:

    print(
        "First conflicting IDs:",
        conflicting_class_ids[:20],
    )


class_unique = (

    class_dataframe

    .drop_duplicates(
        subset=["patientId", "class"]
    )

    .drop_duplicates(
        subset=["patientId"],
        keep="first",
    )

)


exam_manifest = exam_labels.merge(

    class_unique[
        [
            "patientId",
            "class",
        ]
    ],

    on="patientId",

    how="left",

    validate="one_to_one",

)


exam_manifest = exam_manifest.rename(
    columns={
        "patientId": "exam_id",
        "class": "detailed_class",
    }
)


expected_label_map = {
    "Lung Opacity": 1,
    "Normal": 0,
    "No Lung Opacity / Not Normal": 0,
}


exam_manifest["expected_label"] = (

    exam_manifest[
        "detailed_class"
    ]

    .map(
        expected_label_map
    )

)


inconsistent_rows = exam_manifest[

    exam_manifest["expected_label"].notna()

    & (

        exam_manifest["label"]

        != exam_manifest[
            "expected_label"
        ]

    )

]


print("\nLABEL CONSISTENCY")

print("=================")

print(
    "Missing detailed classes:",
    int(
        exam_manifest[
            "detailed_class"
        ].isna().sum()
    ),
)

print(
    "Label/class inconsistencies:",
    len(inconsistent_rows),
)


if len(inconsistent_rows) > 0:

    print(
        inconsistent_rows[
            [
                "exam_id",
                "label",
                "detailed_class",
                "expected_label",
            ]
        ].head(20)
    )


dicom_files = sorted(
    RSNA_IMAGE_DIR.glob("*.dcm")
)


print("\nDICOM FILE AUDIT")

print("================")

print(
    "DICOM files:",
    len(dicom_files),
)


dicom_path_by_exam_id = {
    path.stem: path
    for path in dicom_files
}


label_exam_ids = set(
    exam_manifest["exam_id"]
)


dicom_exam_ids = set(
    dicom_path_by_exam_id.keys()
)


labels_without_images = (

    label_exam_ids

    - dicom_exam_ids

)


images_without_labels = (

    dicom_exam_ids

    - label_exam_ids

)


print(
    "Labelled examinations without DICOM files:",
    len(labels_without_images),
)

print(
    "DICOM files without labels:",
    len(images_without_labels),
)


if labels_without_images:

    print(
        "First labels without images:",
        sorted(
            labels_without_images
        )[:20],
    )


if images_without_labels:

    print(
        "First images without labels:",
        sorted(
            images_without_labels
        )[:20],
    )


metadata_rows = []

dicom_errors = []


print("\nREADING DICOM HEADERS")

print("=====================")


for number, dicom_path in enumerate(
    dicom_files,
    start=1,
):

    try:

        dicom = pydicom.dcmread(
            dicom_path,
            stop_before_pixels=True,
            force=False,
        )


        patient_age_raw = get_dicom_value(
            dicom,
            "PatientAge",
        )


        metadata_rows.append(
            {
                "exam_id": dicom_path.stem,

                "dicom_path": str(
                    dicom_path.resolve()
                ),

                "dicom_patient_id": get_dicom_value(
                    dicom,
                    "PatientID",
                ),

                "study_instance_uid": get_dicom_value(
                    dicom,
                    "StudyInstanceUID",
                ),

                "series_instance_uid": get_dicom_value(
                    dicom,
                    "SeriesInstanceUID",
                ),

                "sop_instance_uid": get_dicom_value(
                    dicom,
                    "SOPInstanceUID",
                ),

                "patient_age_raw": patient_age_raw,

                "patient_age_years": parse_patient_age(
                    patient_age_raw
                ),

                "patient_sex": get_dicom_value(
                    dicom,
                    "PatientSex",
                ),

                "view_position": get_dicom_value(
                    dicom,
                    "ViewPosition",
                ),

                "rows": getattr(
                    dicom,
                    "Rows",
                    None,
                ),

                "columns": getattr(
                    dicom,
                    "Columns",
                    None,
                ),

                "photometric_interpretation": (
                    get_dicom_value(
                        dicom,
                        "PhotometricInterpretation",
                    )
                ),

                "body_part_examined": get_dicom_value(
                    dicom,
                    "BodyPartExamined",
                ),
            }
        )


    except Exception as error:

        dicom_errors.append(
            {
                "dicom_path": str(
                    dicom_path.resolve()
                ),

                "error": repr(error),
            }
        )


    if number % 1000 == 0:

        print(
            f"Read {number:,}/"
            f"{len(dicom_files):,} DICOM headers"
        )


metadata_dataframe = pd.DataFrame(
    metadata_rows
)


metadata_path = (

    MANIFEST_DIR

    / "rsna_dicom_metadata.csv"

)


metadata_dataframe.to_csv(
    metadata_path,
    index=False,
)


print(
    "\nSuccessfully read DICOM headers:",
    len(metadata_dataframe),
)

print(
    "DICOM read errors:",
    len(dicom_errors),
)


if dicom_errors:

    dicom_errors_dataframe = pd.DataFrame(
        dicom_errors
    )

    dicom_errors_dataframe.to_csv(
        TABLES_DIR
        / "rsna_dicom_read_errors.csv",
        index=False,
    )


print("\nVIEW POSITION COUNTS")

print("====================")

print(
    metadata_dataframe[
        "view_position"
    ].value_counts(
        dropna=False
    )
)


print("\nSEX COUNTS")

print("==========")

print(
    metadata_dataframe[
        "patient_sex"
    ].value_counts(
        dropna=False
    )
)


print("\nIMAGE DIMENSION COUNTS")

print("======================")

print(
    metadata_dataframe[
        [
            "rows",
            "columns",
        ]
    ].value_counts(
        dropna=False
    ).head(20)
)


print("\nPHOTOMETRIC INTERPRETATION")

print("==========================")

print(
    metadata_dataframe[
        "photometric_interpretation"
    ].value_counts(
        dropna=False
    )
)


valid_age_values = (

    metadata_dataframe[
        "patient_age_years"
    ]

    .dropna()

)


print("\nAGE SUMMARY")

print("===========")

print(
    "Available age values:",
    len(valid_age_values),
)

print(
    "Missing age values:",
    int(
        metadata_dataframe[
            "patient_age_years"
        ].isna().sum()
    ),
)


if len(valid_age_values) > 0:

    print(
        valid_age_values.describe()
    )

exam_manifest = exam_manifest.merge(

    metadata_dataframe,

    on="exam_id",

    how="left",

    validate="one_to_one",

)


exam_manifest = exam_manifest.drop(
    columns=["expected_label"]
)


initial_manifest_path = (

    MANIFEST_DIR

    / "rsna_exam_manifest_initial.csv"

)


exam_manifest.to_csv(
    initial_manifest_path,
    index=False,
)


print("\nMAPPING JSON AUDIT")

print("==================")


with RSNA_MAPPING_JSON.open(
    "r",
    encoding="utf-8",
) as file:

    mapping_data = json.load(file)


print(
    "Top-level Python type:",
    type(mapping_data).__name__,
)


if isinstance(mapping_data, list):

    print(
        "Number of top-level entries:",
        len(mapping_data),
    )


    if mapping_data:

        first_mapping_entry = mapping_data[0]

        print(
            "\nFirst mapping entry:"
        )

        print(
            json.dumps(
                first_mapping_entry,
                indent=2,
            )[:5000]
        )


elif isinstance(mapping_data, dict):

    print(
        "Number of top-level keys:",
        len(mapping_data),
    )


    first_mapping_keys = list(
        mapping_data.keys()
    )[:10]


    print(
        "First top-level keys:",
        first_mapping_keys,
    )


    if first_mapping_keys:

        first_mapping_key = (
            first_mapping_keys[0]
        )

        print(
            "\nFirst mapping value:"
        )

        print(
            json.dumps(
                mapping_data[
                    first_mapping_key
                ],
                indent=2,
            )[:5000]
        )


json_key_counter = Counter()

count_json_keys(
    mapping_data,
    json_key_counter,
)


mapping_key_dataframe = pd.DataFrame(
    [
        {
            "json_key": key,
            "occurrences": count,
        }

        for key, count
        in json_key_counter.most_common()
    ]
)


mapping_key_path = (

    TABLES_DIR

    / "rsna_mapping_json_key_counts.csv"

)


mapping_key_dataframe.to_csv(
    mapping_key_path,
    index=False,
)


print("\nMOST COMMON MAPPING JSON KEYS")

print("=============================")

print(
    mapping_key_dataframe.head(30).to_string(
        index=False
    )
)

audit_report = {
    "raw_label_rows": int(
        len(labels_dataframe)
    ),

    "unique_examinations": int(
        len(exam_manifest)
    ),

    "negative_examinations": int(
        (
            exam_manifest["label"] == 0
        ).sum()
    ),

    "positive_examinations": int(
        (
            exam_manifest["label"] == 1
        ).sum()
    ),

    "multiple_box_examinations": int(
        (
            exam_manifest["box_count"] > 1
        ).sum()
    ),

    "maximum_boxes_per_examination": int(
        exam_manifest["box_count"].max()
    ),

    "dicom_file_count": int(
        len(dicom_files)
    ),

    "labels_without_images": int(
        len(labels_without_images)
    ),

    "images_without_labels": int(
        len(images_without_labels)
    ),

    "dicom_read_errors": int(
        len(dicom_errors)
    ),

    "missing_detailed_classes": int(
        exam_manifest[
            "detailed_class"
        ].isna().sum()
    ),

    "label_class_inconsistencies": int(
        len(inconsistent_rows)
    ),

    "view_position_counts": {
        str(key): int(value)

        for key, value
        in metadata_dataframe[
            "view_position"
        ].value_counts(
            dropna=False
        ).items()
    },

    "sex_counts": {
        str(key): int(value)

        for key, value
        in metadata_dataframe[
            "patient_sex"
        ].value_counts(
            dropna=False
        ).items()
    },

    "mapping_top_level_type": (
        type(mapping_data).__name__
    ),

    "mapping_top_level_size": (
        len(mapping_data)
        if hasattr(
            mapping_data,
            "__len__",
        )
        else None
    ),
}


audit_report_path = (

    TABLES_DIR

    / "rsna_dataset_audit.json"

)


with audit_report_path.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        audit_report,
        file,
        indent=2,
    )


print("\nGENERATED FILES")

print("===============")

print(
    "DICOM metadata:",
    metadata_path,
)

print(
    "Initial examination manifest:",
    initial_manifest_path,
)

print(
    "Mapping key table:",
    mapping_key_path,
)

print(
    "Dataset audit report:",
    audit_report_path,
)


print("\nRSNA DATASET AUDIT COMPLETED")