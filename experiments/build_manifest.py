import json

import pandas as pd

from sklearn.model_selection import (
    StratifiedGroupKFold,
)

from config import (
    MANIFEST_DIR,
    RSNA_MAPPING_JSON,
    TABLES_DIR,
)


SPLIT_SEED = 42

AGE_MINIMUM = 0
AGE_MAXIMUM = 120

INITIAL_MANIFEST_PATH = (
    MANIFEST_DIR
    / "rsna_exam_manifest_initial.csv"
)

FULL_MANIFEST_PATH = (
    MANIFEST_DIR
    / "full_manifest.csv"
)

TRAIN_MANIFEST_PATH = (
    MANIFEST_DIR
    / "train.csv"
)

VALIDATION_MANIFEST_PATH = (
    MANIFEST_DIR
    / "validation.csv"
)

TEST_MANIFEST_PATH = (
    MANIFEST_DIR
    / "test.csv"
)


def choose_best_grouped_fold(
    dataframe,
    number_of_splits,
    target_fraction,
    random_seed,
):

    splitter = StratifiedGroupKFold(
        n_splits=number_of_splits,
        shuffle=True,
        random_state=random_seed,
    )


    overall_positive_fraction = (
        dataframe["label"].mean()
    )


    overall_class_distribution = (
        dataframe["detailed_class"]
        .value_counts(normalize=True)
    )


    best_result = None


    split_iterator = splitter.split(
        X=dataframe,
        y=dataframe["label"],
        groups=dataframe[
            "original_patient_id"
        ],
    )


    for fold_number, (
        remaining_indices,
        holdout_indices,
    ) in enumerate(split_iterator):

        remaining_dataframe = (
            dataframe
            .iloc[remaining_indices]
            .copy()
        )


        holdout_dataframe = (
            dataframe
            .iloc[holdout_indices]
            .copy()
        )


        holdout_fraction = (
            len(holdout_dataframe)
            / len(dataframe)
        )


        holdout_positive_fraction = (
            holdout_dataframe[
                "label"
            ].mean()
        )


        holdout_class_distribution = (
            holdout_dataframe[
                "detailed_class"
            ]
            .value_counts(
                normalize=True
            )
            .reindex(
                overall_class_distribution.index,
                fill_value=0.0,
            )
        )


        size_error = abs(
            holdout_fraction
            - target_fraction
        )


        prevalence_error = abs(
            holdout_positive_fraction
            - overall_positive_fraction
        )


        detailed_class_error = (
            (
                holdout_class_distribution
                - overall_class_distribution
            )
            .abs()
            .mean()
        )


        score = (
            2.0 * size_error
            + prevalence_error
            + detailed_class_error
        )


        candidate = {
            "score": score,
            "fold_number": fold_number,
            "remaining": remaining_dataframe,
            "holdout": holdout_dataframe,
            "holdout_fraction": (
                holdout_fraction
            ),
            "holdout_positive_fraction": (
                holdout_positive_fraction
            ),
        }


        if (
            best_result is None
            or score
            < best_result["score"]
        ):

            best_result = candidate


    if best_result is None:

        raise RuntimeError(
            "No valid grouped fold was created."
        )


    return best_result


MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("\nINPUT FILE CHECK")
print("================")


required_paths = [
    INITIAL_MANIFEST_PATH,
    RSNA_MAPPING_JSON,
]


for path in required_paths:

    print(
        "FOUND" if path.exists()
        else "MISSING",
        path,
    )


missing_paths = [
    path
    for path in required_paths
    if not path.exists()
]


if missing_paths:

    raise FileNotFoundError(
        "\n".join(
            str(path)
            for path in missing_paths
        )
    )


initial_manifest = pd.read_csv(
    INITIAL_MANIFEST_PATH,
    dtype={
        "exam_id": str,
    },
)


print("\nINITIAL MANIFEST")
print("================")

print(
    "Examinations:",
    len(initial_manifest),
)

print(
    "Unique examination IDs:",
    initial_manifest[
        "exam_id"
    ].nunique(),
)


if (
    initial_manifest["exam_id"]
    .duplicated()
    .any()
):

    raise ValueError(
        "The initial manifest contains "
        "duplicate examination IDs."
    )


with RSNA_MAPPING_JSON.open(
    "r",
    encoding="utf-8",
) as file:

    mapping_json = json.load(file)


if not isinstance(
    mapping_json,
    list,
):

    raise TypeError(
        "Expected the mapping JSON "
        "to contain a top-level list."
    )


mapping_dataframe = pd.DataFrame(
    mapping_json
)


required_mapping_columns = {
    "img_id",
    "subset_img_id",
    "subset_group",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
}


missing_mapping_columns = (
    required_mapping_columns
    - set(
        mapping_dataframe.columns
    )
)


if missing_mapping_columns:

    raise ValueError(
        "Mapping JSON is missing columns: "
        + str(
            sorted(
                missing_mapping_columns
            )
        )
    )


print("\nMAPPING FILE")
print("============")

print(
    "Mapping entries:",
    len(mapping_dataframe),
)

print(
    "Unique RSNA examination IDs:",
    mapping_dataframe[
        "subset_img_id"
    ].nunique(),
)

print(
    "Unique original NIH images:",
    mapping_dataframe[
        "img_id"
    ].nunique(),
)


mapping_duplicate_count = int(
    mapping_dataframe[
        "subset_img_id"
    ]
    .duplicated()
    .sum()
)


print(
    "Duplicate subset_img_id values:",
    mapping_duplicate_count,
)


if mapping_duplicate_count > 0:

    raise ValueError(
        "The mapping contains duplicate "
        "RSNA examination identifiers."
    )


mapping_dataframe = (
    mapping_dataframe[
        [
            "subset_img_id",
            "img_id",
            "subset_group",
            "subset_init_label",
            "orig_labels",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "studyId",
            "seriesId",
            "instanceId",
        ]
    ]
    .copy()
)


mapping_dataframe = (
    mapping_dataframe.rename(
        columns={
            "subset_img_id": "exam_id",
            "img_id": (
                "original_nih_image"
            ),
            "StudyInstanceUID": (
                "mapping_study_instance_uid"
            ),
            "SeriesInstanceUID": (
                "mapping_series_instance_uid"
            ),
            "SOPInstanceUID": (
                "mapping_sop_instance_uid"
            ),
        }
    )
)


mapping_dataframe["exam_id"] = (
    mapping_dataframe[
        "exam_id"
    ].astype(str)
)


mapping_dataframe[
    "original_nih_image"
] = (
    mapping_dataframe[
        "original_nih_image"
    ].astype(str)
)

mapping_dataframe[
    "original_patient_id"
] = (
    mapping_dataframe[
        "original_nih_image"
    ]
    .str.split(
        "_",
        n=1,
    )
    .str[0]
)


missing_patient_ids = int(
    mapping_dataframe[
        "original_patient_id"
    ]
    .isna()
    .sum()
)


empty_patient_ids = int(
    (
        mapping_dataframe[
            "original_patient_id"
        ]
        .astype(str)
        .str.len()
        == 0
    ).sum()
)


print(
    "Missing derived patient IDs:",
    missing_patient_ids,
)

print(
    "Empty derived patient IDs:",
    empty_patient_ids,
)


if (
    missing_patient_ids > 0
    or empty_patient_ids > 0
):

    raise ValueError(
        "Could not derive all original "
        "patient identifiers."
    )


full_manifest = (
    initial_manifest.merge(
        mapping_dataframe,
        on="exam_id",
        how="left",
        validate="one_to_one",
    )
)


missing_mapping_count = int(
    full_manifest[
        "original_nih_image"
    ]
    .isna()
    .sum()
)


print("\nMAPPING MERGE")
print("=============")

print(
    "Labelled examinations:",
    len(full_manifest),
)

print(
    "Examinations without mapping:",
    missing_mapping_count,
)


if missing_mapping_count > 0:

    missing_mapping_examples = (
        full_manifest.loc[
            full_manifest[
                "original_nih_image"
            ].isna(),
            "exam_id",
        ]
        .head(20)
        .tolist()
    )


    print(
        "First missing mapping IDs:",
        missing_mapping_examples,
    )


    raise ValueError(
        "Some labelled examinations "
        "were not found in the mapping."
    )


patient_exam_counts = (
    full_manifest
    .groupby(
        "original_patient_id"
    )
    .size()
)


print("\nPATIENT GROUP AUDIT")
print("===================")

print(
    "Unique original patients:",
    full_manifest[
        "original_patient_id"
    ].nunique(),
)

print(
    "Patients with more than one examination:",
    int(
        (
            patient_exam_counts > 1
        ).sum()
    ),
)

print(
    "Largest number of examinations "
    "for one patient:",
    int(
        patient_exam_counts.max()
    ),
)


duplicate_original_images = int(
    full_manifest[
        "original_nih_image"
    ]
    .duplicated()
    .sum()
)


duplicate_sop_uids = int(
    full_manifest[
        "sop_instance_uid"
    ]
    .duplicated()
    .sum()
)


print("\nIDENTIFIER AUDIT")
print("================")

print(
    "Duplicate original NIH images:",
    duplicate_original_images,
)

print(
    "Duplicate DICOM SOP Instance UIDs:",
    duplicate_sop_uids,
)


if duplicate_original_images > 0:

    raise ValueError(
        "Duplicate original NIH image "
        "identifiers were found."
    )


if duplicate_sop_uids > 0:

    raise ValueError(
        "Duplicate SOP Instance UIDs "
        "were found."
    )


study_uid_matches = (
    full_manifest[
        "study_instance_uid"
    ].astype(str)
    ==
    full_manifest[
        "mapping_study_instance_uid"
    ].astype(str)
)


series_uid_matches = (
    full_manifest[
        "series_instance_uid"
    ].astype(str)
    ==
    full_manifest[
        "mapping_series_instance_uid"
    ].astype(str)
)


sop_uid_matches = (
    full_manifest[
        "sop_instance_uid"
    ].astype(str)
    ==
    full_manifest[
        "mapping_sop_instance_uid"
    ].astype(str)
)


print("\nDICOM-MAPPING UID CHECK")
print("=======================")

print(
    "Study UID mismatches:",
    int(
        (~study_uid_matches).sum()
    ),
)

print(
    "Series UID mismatches:",
    int(
        (~series_uid_matches).sum()
    ),
)

print(
    "SOP UID mismatches:",
    int(
        (~sop_uid_matches).sum()
    ),
)


if (
    (~study_uid_matches).any()
    or (~series_uid_matches).any()
    or (~sop_uid_matches).any()
):

    raise ValueError(
        "The DICOM UIDs and mapping "
        "UIDs are inconsistent."
    )


full_manifest[
    "age_metadata_outlier"
] = (
    full_manifest[
        "patient_age_years"
    ].notna()
    &
    (
        (
            full_manifest[
                "patient_age_years"
            ]
            < AGE_MINIMUM
        )
        |
        (
            full_manifest[
                "patient_age_years"
            ]
            > AGE_MAXIMUM
        )
    )
)


full_manifest[
    "patient_age_years_clean"
] = (
    full_manifest[
        "patient_age_years"
    ].where(
        ~full_manifest[
            "age_metadata_outlier"
        ]
    )
)


print("\nAGE METADATA CLEANING")
print("=====================")

print(
    "Age metadata outliers:",
    int(
        full_manifest[
            "age_metadata_outlier"
        ].sum()
    ),
)

print(
    "Clean age values:",
    int(
        full_manifest[
            "patient_age_years_clean"
        ].notna()
        .sum()
    ),
)


age_outliers = full_manifest[
    full_manifest[
        "age_metadata_outlier"
    ]
]


if len(age_outliers) > 0:

    age_outliers[
        [
            "exam_id",
            "original_patient_id",
            "patient_age_raw",
            "patient_age_years",
        ]
    ].to_csv(
        TABLES_DIR
        / "rsna_age_metadata_outliers.csv",
        index=False,
    )

test_selection = (
    choose_best_grouped_fold(
        dataframe=full_manifest,
        number_of_splits=7,
        target_fraction=0.15,
        random_seed=SPLIT_SEED,
    )
)


development_manifest = (
    test_selection[
        "remaining"
    ].copy()
)


test_manifest = (
    test_selection[
        "holdout"
    ].copy()
)


actual_test_fraction = (
    len(test_manifest)
    / len(full_manifest)
)


conditional_validation_fraction = (
    0.15
    / (
        1.0
        - actual_test_fraction
    )
)


validation_selection = (
    choose_best_grouped_fold(
        dataframe=development_manifest,
        number_of_splits=6,
        target_fraction=(
            conditional_validation_fraction
        ),
        random_seed=SPLIT_SEED + 1,
    )
)


train_manifest = (
    validation_selection[
        "remaining"
    ].copy()
)


validation_manifest = (
    validation_selection[
        "holdout"
    ].copy()
)


train_manifest["split"] = "train"

validation_manifest[
    "split"
] = "validation"

test_manifest["split"] = "test"


train_patients = set(
    train_manifest[
        "original_patient_id"
    ]
)


validation_patients = set(
    validation_manifest[
        "original_patient_id"
    ]
)


test_patients = set(
    test_manifest[
        "original_patient_id"
    ]
)


train_validation_overlap = (
    train_patients
    & validation_patients
)


train_test_overlap = (
    train_patients
    & test_patients
)


validation_test_overlap = (
    validation_patients
    & test_patients
)


print("\nPATIENT LEAKAGE CHECK")
print("=====================")

print(
    "Train-validation overlap:",
    len(
        train_validation_overlap
    ),
)

print(
    "Train-test overlap:",
    len(
        train_test_overlap
    ),
)

print(
    "Validation-test overlap:",
    len(
        validation_test_overlap
    ),
)


if (
    train_validation_overlap
    or train_test_overlap
    or validation_test_overlap
):

    raise RuntimeError(
        "Patient leakage was found "
        "between data splits."
    )


full_manifest = pd.concat(
    [
        train_manifest,
        validation_manifest,
        test_manifest,
    ],
    ignore_index=True,
)


split_order = pd.CategoricalDtype(
    categories=[
        "train",
        "validation",
        "test",
    ],
    ordered=True,
)


full_manifest["split"] = (
    full_manifest[
        "split"
    ].astype(
        split_order
    )
)


full_manifest = (
    full_manifest.sort_values(
        [
            "split",
            "original_patient_id",
            "exam_id",
        ]
    )
    .reset_index(drop=True)
)


train_manifest = (
    full_manifest[
        full_manifest[
            "split"
        ]
        == "train"
    ]
    .copy()
)


validation_manifest = (
    full_manifest[
        full_manifest[
            "split"
        ]
        == "validation"
    ]
    .copy()
)


test_manifest = (
    full_manifest[
        full_manifest[
            "split"
        ]
        == "test"
    ]
    .copy()
)


for dataframe in [
    full_manifest,
    train_manifest,
    validation_manifest,
    test_manifest,
]:

    dataframe["split"] = (
        dataframe[
            "split"
        ].astype(str)
    )


full_manifest.to_csv(
    FULL_MANIFEST_PATH,
    index=False,
)


train_manifest.to_csv(
    TRAIN_MANIFEST_PATH,
    index=False,
)


validation_manifest.to_csv(
    VALIDATION_MANIFEST_PATH,
    index=False,
)


test_manifest.to_csv(
    TEST_MANIFEST_PATH,
    index=False,
)


summary_rows = []


for split_name, dataframe in [
    ("train", train_manifest),
    (
        "validation",
        validation_manifest,
    ),
    ("test", test_manifest),
]:

    positive_count = int(
        (
            dataframe["label"] == 1
        ).sum()
    )


    negative_count = int(
        (
            dataframe["label"] == 0
        ).sum()
    )


    summary_rows.append(
        {
            "split": split_name,
            "images": len(dataframe),
            "percentage_of_dataset": (
                100.0
                * len(dataframe)
                / len(full_manifest)
            ),
            "unique_patients": (
                dataframe[
                    "original_patient_id"
                ].nunique()
            ),
            "positive": positive_count,
            "negative": negative_count,
            "positive_fraction": (
                positive_count
                / len(dataframe)
            ),
            "PA": int(
                (
                    dataframe[
                        "view_position"
                    ]
                    == "PA"
                ).sum()
            ),
            "AP": int(
                (
                    dataframe[
                        "view_position"
                    ]
                    == "AP"
                ).sum()
            ),
        }
    )


split_summary = pd.DataFrame(
    summary_rows
)


split_summary.to_csv(
    TABLES_DIR
    / "rsna_split_summary.csv",
    index=False,
)


detailed_class_summary = pd.crosstab(
    full_manifest["split"],
    full_manifest[
        "detailed_class"
    ],
)


detailed_class_summary.to_csv(
    TABLES_DIR
    / "rsna_split_detailed_classes.csv"
)


view_summary = pd.crosstab(
    full_manifest["split"],
    full_manifest[
        "view_position"
    ],
)


view_summary.to_csv(
    TABLES_DIR
    / "rsna_split_view_positions.csv"
)

print("\nFINAL SPLIT SUMMARY")
print("===================")

print(
    split_summary.to_string(
        index=False
    )
)


print("\nDETAILED CLASS COUNTS")
print("=====================")

print(
    detailed_class_summary.to_string()
)


print("\nVIEW POSITION COUNTS")
print("====================")

print(
    view_summary.to_string()
)


print("\nGENERATED MANIFESTS")
print("===================")

print(
    "Full manifest:",
    FULL_MANIFEST_PATH,
)

print(
    "Training manifest:",
    TRAIN_MANIFEST_PATH,
)

print(
    "Validation manifest:",
    VALIDATION_MANIFEST_PATH,
)

print(
    "Test manifest:",
    TEST_MANIFEST_PATH,
)

print(
    "Split summary:",
    TABLES_DIR
    / "rsna_split_summary.csv",
)


print(
    "\nPATIENT-AWARE MANIFEST "
    "BUILD COMPLETED"
)