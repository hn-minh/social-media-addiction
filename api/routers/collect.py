import logging
import pandas as pd
from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, exc
import random
import time
import os
from dotenv import load_dotenv
from api.schemas.user_report import UserReport
from src import config

logger = logging.getLogger(__name__)
load_dotenv()
DATASET_URL = os.getenv('DATASET_URL')

router = APIRouter(
    prefix="/collect",
    tags=["Data Collection"]
)

@router.post("/")
def collect_user_data(data: UserReport):
    logger.info("Received new user report data.")
    try:
        input_dict = data.model_dump()
        generated_id = int(time.time()) * 100 + random.randint(1, 99)
        
        final_data = {"Student_ID": generated_id}
        final_data.update(input_dict)
        
        df_new = pd.DataFrame([final_data])
        
        engine = create_engine(DATASET_URL)
        
        df_new.to_sql('users_behavior', con=engine, if_exists='append', index=False)
        
        logger.info(f"Successfully saved record {generated_id} to cloud database.")
        
        return {
            "status": "success",
            "student_id": generated_id,
            "message": "Thank you! Data has been securely contributed to the cloud system."
        }
        
    except exc.SQLAlchemyError as sql_e:
        logger.error(f"Database error while saving data: {str(sql_e)}")
        raise HTTPException(status_code=500, detail="Database connection or write error.")
        
    except Exception as e:
        logger.error(f"System error during data collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")