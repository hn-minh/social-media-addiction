from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "Students Social Media Addiction.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

TARGET_COLUMN = "Addicted_Score"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_SAMPLES_REQUIRED = 5


MLFLOW_EXPERIMENT_NAME = "social_media_addiction_classification"
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR}/mlflow.db".replace("\\", "/")

ASIAN_COUNTRIES = ['Bangladesh', 'India', 'China', 'Japan', 'South Korea', 'Singapore', 'Malaysia', 
                    'Thailand', 'Vietnam', 'Philippines', 'Indonesia', 'Taiwan', 'Hong Kong']
EUROPEAN_COUNTRIES = ['UK', 'Germany', 'France', 'Spain', 'Italy', 'Sweden', 'Norway', 'Denmark', 
                        'Netherlands', 'Belgium', 'Switzerland', 'Austria', 'Portugal', 'Greece', 'Ireland']
AMERICAN_COUNTRIES = ['USA', 'Canada', 'Brazil', 'Mexico', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela']