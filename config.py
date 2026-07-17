from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


RSNA_RAW_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "rsna"
    / "raw"
)

RSNA_IMAGE_DIR = (
    RSNA_RAW_DIR
    / "stage_2_train_images"
)

RSNA_LABELS_CSV = (
    RSNA_RAW_DIR
    / "stage_2_train_labels.csv"
)

RSNA_CLASS_INFO_CSV = (
    RSNA_RAW_DIR
    / "stage_2_detailed_class_info.csv"
)

RSNA_MAPPING_JSON = (
    RSNA_RAW_DIR
    / "pneumonia-challenge-dataset-mappings_2018.json"
)


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 1


DATA_DIR = PROJECT_ROOT / "data"

MANIFEST_DIR = DATA_DIR / "manifests"

PARTITION_DIR = DATA_DIR / "partitions"

CACHE_DIR = (
    DATA_DIR
    / "cache"
    / f"rsna_{IMAGE_HEIGHT}x{IMAGE_WIDTH}"
)

RESULTS_DIR = PROJECT_ROOT / "results"

TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR = RESULTS_DIR / "figures"



BATCH_SIZE = 32
LEARNING_RATE = 0.001
CENTRALIZED_EPOCHS = 20

NUM_CLIENTS = 5
ROUNDS = 20
LOCAL_EPOCHS = 1

FEDPROX_MU = 0.01

SEEDS = [11, 22, 33]

PARTITION_SEEDS = SEEDS

DIRICHLET_ALPHAS = {
    "alpha_05": 0.5,
    "alpha_01": 0.1,
}

MIN_CLIENT_IMAGES = 200