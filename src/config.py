from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

HEALTHY_DATA_PATH = DATA_DIR / "healthy_data.csv"
IBD_DATA_PATH = DATA_DIR / "ibd_data.csv"
MS_DATA_PATH = DATA_DIR / "ms_data.csv"
RA_DATA_PATH = DATA_DIR / "ra_data.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FEATURES = 20

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)