import pandas as pd
from sqlalchemy import create_engine
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()
DATASET_URL = os.getenv("DATASET_URL")
CSV_PATH = "data/Students Social Media Addiction.csv"

def upload_csv_to_cloud():
    try:
        logger.info("Reading CSV file")
        df = pd.read_csv(CSV_PATH)
        
        logger.info("Connecting to Database")
        engine = create_engine(DATASET_URL)
        
        df.to_sql('users_behavior', con=engine, if_exists='replace', index=False)
        
        logger.info("Pushed data successfully.")
    except Exception as e:
        logger.error(f"Error when push data: {e}")

if __name__ == "__main__":
    upload_csv_to_cloud()