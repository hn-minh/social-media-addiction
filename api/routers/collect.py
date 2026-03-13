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
        
        save_dir = config.DATA_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "collected_data.csv"
        
        if not os.path.exists(save_path):
            df_new.to_csv(save_path, index=False)
            logger.info(f"Đã tạo file mới và lưu dữ liệu với ID: {generated_id}")
        else:
            df_new.to_csv(save_path, mode='a', header=False, index=False)
            logger.info(f"Đã append dữ liệu thành công (ID: {generated_id}).")
            
        return {
            "status": "success",
            "student_id": generated_id,
            "message": "Cảm ơn bạn! Dữ liệu đã được lưu trữ an toàn."
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lưu dữ liệu thu thập: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Không thể lưu dữ liệu: {str(e)}")