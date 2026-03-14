import logging
import pandas as pd
from fastapi import APIRouter, HTTPException
import os
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
        
        save_path = config.DATA_PATH 
        
        if not os.path.exists(save_path):
            df_new.to_csv(save_path, index=False)
            logger.info(f"Đã tạo lại file dataset gốc và lưu dữ liệu với ID: {generated_id}")
        else:
            existing_cols = pd.read_csv(save_path, nrows=0).columns
            
            for col in existing_cols:
                if col not in df_new.columns:
                    df_new[col] = None
                    
            df_new = df_new[existing_cols]
            
            df_new.to_csv(save_path, mode='a', header=False, index=False)
            logger.info(f"Đã ghi trực tiếp vào dataset gốc thành công (ID: {generated_id}).")
            
        return {
            "status": "success",
            "student_id": generated_id,
            "message": "Cảm ơn bạn! Dữ liệu đã được đóng góp."
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lưu dữ liệu thu thập: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Không thể lưu dữ liệu: {str(e)}")