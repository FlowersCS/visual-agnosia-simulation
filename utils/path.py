import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
DATA_ROOT_PATH = PROJECT_ROOT
print(DATA_ROOT_PATH)
sys.path.append(str(PROJECT_ROOT))

# Data related
DATASET_PATH = DATA_ROOT_PATH / "dataset"
EXPERIMENT_DIR = DATA_ROOT_PATH / "experiments"