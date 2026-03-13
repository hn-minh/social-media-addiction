import pandas as pd
import logging
from pathlib import Path
import sys
from src import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_data(file_path: Path) -> pd.DataFrame:
    logger.info(f"Attempting to read data from: {file_path}")
    
    try:
        if not file_path.exists():
            logger.error(f"File not found error. Please check the path: {file_path}")
            raise FileNotFoundError(f"Data file is missing at {file_path}")

        df = pd.read_csv(file_path)
        
        if df.empty:
            logger.warning(f"The file was read successfully, but the dataset is empty: {file_path}")
            raise ValueError("The loaded dataset contains no records.")
            
        if config.TARGET_COLUMN not in df.columns:
            logger.error(f"Target column '{config.TARGET_COLUMN}' is missing from the dataset.")
            raise KeyError(f"Missing target column: {config.TARGET_COLUMN}")

        logger.info(f"Data ingestion successful. Dataset shape: {df.shape}")
        
        missing_values = df.isnull().sum().sum()
        logger.info(f"Total missing values found in the dataset: {missing_values}")
        
        return df

    except Exception as e:
        logger.critical(f"Data ingestion failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        df_raw = load_data(config.DATA_PATH)
        logger.info("Test passed. Data is ready for preprocessing.")
    except Exception as e:
        logger.error("Test failed. Please check the logs above.")