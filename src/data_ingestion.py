import pandas as pd
import logging
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()
DATASET_URL = os.getenv("DATASET_URL")

def load_data() -> pd.DataFrame:
    logger.info("Bắt đầu kết nối tới Cloud PostgreSQL để tải dữ liệu...")
    try:
        engine = create_engine(DATASET_URL)
        
        query = "SELECT * FROM users_behavior"
        
        df = pd.read_sql(query, con=engine)
        
        logger.info(f"✅ Đã tải thành công {len(df)} dòng dữ liệu từ mây về RAM.")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi tải dữ liệu từ Cloud Database: {str(e)}")
        raise e