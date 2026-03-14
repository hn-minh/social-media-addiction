import logging
import pandas as pd
from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, exc
import uuid

from api.schemas.user_report import UserReport
from src import config


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/collect",
    tags=["Data Collection"]
)

@router.post("/")
def collect_user_data(data: UserReport):
    logger.info("Nhận được dữ liệu phản hồi mới từ người dùng.")
    try:
        input_dict = data.model_dump()
        generated_id = f"{uuid.uuid4().hex[:8].upper()}"
        
        final_data = {"Student_ID": generated_id}
        final_data.update(input_dict)
        
        df_new = pd.DataFrame([final_data])
        
        engine = create_engine(config.CLOUD_DB_URL)
        
        df_new.to_sql('users_behavior', con=engine, if_exists='append', index=False)
        
        logger.info(f"Đã ghi bản ghi {generated_id} trực tiếp vào Cloud Database thành công.")
        
        return {
            "status": "success",
            "student_id": generated_id,
            "message": "Cảm ơn bạn! Dữ liệu đã được đóng góp an toàn lên hệ thống đám mây."
        }
        
    except exc.SQLAlchemyError as sql_e:
        logger.error(f"❌ Lỗi Database khi lưu dữ liệu: {str(sql_e)}")
        raise HTTPException(status_code=500, detail="Lỗi kết nối hoặc ghi vào Database.")
        
    except Exception as e:
        logger.error(f"❌ Lỗi hệ thống khi xử lý dữ liệu thu thập: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Không thể lưu dữ liệu: {str(e)}")