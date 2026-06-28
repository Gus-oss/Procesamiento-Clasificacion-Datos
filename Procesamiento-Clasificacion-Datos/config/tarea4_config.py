from pathlib import Path

# Raíz del proyecto — subo dos niveles desde config/
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Dataset ───────────────────────────────────────────────────────────────────
# Descargo MVTec AD desde: https://www.mvtec.com/company/research/datasets/mvtec-ad
# y lo extraigo aquí
MVTEC_ROOT = BASE_DIR / "data" / "raw" / "mvtec"
OUTPUT_DIR  = BASE_DIR / "data" / "processed" / "tarea4"

CATEGORIES = [
    "carpet", "grid", "leather", "tile", "wood",          # texturas
    "bottle", "cable", "capsule", "hazelnut", "metal_nut", # objetos
    "pill", "screw", "toothbrush", "transistor", "zipper",
]

# ── Preprocesamiento OpenCV ────────────────────────────────────────────────────
IMAGE_SIZE      = (256, 256)   # tamaño para extracción de features
CNN_INPUT_SIZE  = (224, 224)   # tamaño de entrada a la red
GAUSSIAN_KERNEL = (5, 5)
CANNY_T1, CANNY_T2 = 50, 150
HARRIS_BLOCK, HARRIS_K = 2, 0.04
ORB_KEYPOINTS   = 500
HIST_BINS       = 32

# ── Entrenamiento ─────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
VAL_SPLIT   = 0.2
SEED        = 42
NUM_WORKERS = 4
EARLY_STOP_PATIENCE  = 10
LR_PATIENCE          = 5
LR_FACTOR            = 0.5

# ── Optuna ────────────────────────────────────────────────────────────────────
N_TRIALS      = 20
TRIAL_EPOCHS  = 10
LR_RANGE      = (1e-5, 1e-2)
DROPOUT_RANGE = (0.1, 0.5)
OPTIMIZERS    = ["Adam", "AdamW", "SGD"]

# ── PySpark ───────────────────────────────────────────────────────────────────
SPARK_APP_NAME      = "MVTec_QualityControl"
SPARK_DRIVER_MEM    = "4g"
SPARK_PARTITIONS    = 8
