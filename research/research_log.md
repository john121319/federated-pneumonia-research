## Stage 1 — RSNA dataset audit

### Objective

Audit the raw RSNA dataset and verify image-label consistency before
creating model-development partitions.

### Results

- Raw annotation rows: 30,227
- Unique examinations: 26,684
- Positive examinations: 6,012
- Negative examinations: 20,672
- Lung Opacity examinations: 6,012
- Normal examinations: 8,851
- No Lung Opacity / Not Normal examinations: 11,821
- Examinations with multiple bounding boxes: 3,398
- Maximum bounding boxes in one examination: 4
- PA images: 14,511
- AP images: 12,173
- Male records: 15,166
- Female records: 11,518
- DICOM read errors: 0
- Labels without images: 0
- Images without labels: 0
- Missing detailed classes: 0
- Label inconsistencies: 0

### Metadata observations

All images were 1024 × 1024 pixels and used MONOCHROME2
photometric interpretation. Age metadata were available for all
examinations, but at least one implausible value was observed, with a
maximum reported age of 155 years. Implausible age values will be
retained in the raw metadata and marked as outliers in the processed
manifest. Age will not be used as a model input.

### Mapping observations

The RSNA-to-NIH mapping contained 30,000 entries. The subset_img_id
field represents the randomized RSNA examination identifier, while
img_id stores the corresponding original NIH image filename. Original
patient groups will be derived from the filename prefix before the
underscore.

### Decision

Construct patient-grouped training, validation and test partitions so
that no original NIH patient appears in more than one partition.

## Stage 2 — Patient-aware manifest

### Objective

Merge the labelled RSNA examinations with the original NIH image
mapping and construct leakage-resistant training, validation and test
partitions.

### Results

- Labelled examinations: 26,684
- Unique original patients: 11,452
- Patients with multiple examinations: 4,355
- Maximum examinations for one patient: 65
- Examinations without mapping: 0
- Duplicate original NIH images: 0
- Duplicate DICOM SOP Instance UIDs: 0
- DICOM-to-mapping UID mismatches: 0
- Age metadata outliers: 5

### Final split

| Split | Images | Patients | Positive | Negative | Positive fraction |
|---|---:|---:|---:|---:|---:|
| Training | 18,981 | 8,148 | 4,289 | 14,692 | 0.2260 |
| Validation | 3,841 | 1,658 | 846 | 2,995 | 0.2203 |
| Test | 3,862 | 1,646 | 877 | 2,985 | 0.2271 |

### Leakage checks

- Training-validation patient overlap: 0
- Training-test patient overlap: 0
- Validation-test patient overlap: 0

### Decision

Use the patient-aware partitions for every centralized, local-only,
FedAvg and FedProx experiment. The validation set will be used for
model selection, and the test set will remain untouched until final
evaluation.