from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
RESULT_DIR = ROOT_DIR / "results"

FIGURE_DIR = RESULT_DIR / "figures"
REPORT_DIR = RESULT_DIR / "reports"
METRIC_DIR = RESULT_DIR / "metrics"

DATA_PATH = DATA_DIR / "diabetes.csv"

# =============================================================================
# Dataset
# =============================================================================

TARGET_COLUMN = "Outcome"

TEST_SIZE = 0.2

RANDOM_STATE = 42

# =============================================================================
# Cross Validation
# =============================================================================

CV = 5

SCORING = "recall"

N_JOBS = -1

# =============================================================================
# SVM Hyperparameters
# =============================================================================

PARAM_GRID = {

    "C": [0.1, 1, 10],

    "kernel": [

        "linear",
        "poly",
        "rbf"

    ],

    "degree": [

        1,
        2,
        3

    ],

    "gamma": [

        "scale",
        "auto",
        1

    ]

}

# =============================================================================
# Log Transform Columns
# =============================================================================

LOG_COLUMNS = [

    "Pregnancies",

    "SkinThickness",

    "Insulin",

    "DiabetesPedigreeFunction",

    "Age"

]

# =============================================================================
# Saved Models
# =============================================================================

STANDARD_MODEL = MODEL_DIR / "svm_standard.pkl"

MINMAX_MODEL = MODEL_DIR / "svm_minmax.pkl"

LOG_STANDARD_MODEL = MODEL_DIR / "svm_log_standard.pkl"

LOG_MINMAX_MODEL = MODEL_DIR / "svm_log_minmax.pkl"