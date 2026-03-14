import pandas as pd
import logging
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()
DATASET_URL = os.getenv("DATASET_URL")

def load_data() -> pd.DataFrame:
    logger.info("Connecting to Cloud PostgreSQL to load data.")
    try:
        engine = create_engine(DATASET_URL)
        
        query = "SELECT * FROM users_behavior"
        
        df = pd.read_sql(query, con=engine)
        
        logger.info(f"Successfully loaded {len(df)} rows from cloud to memory.")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data from cloud database: {str(e)}")
        raise e