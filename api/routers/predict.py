import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from api.schemas.user_input import UserInput
from src.preprocessing import preprocess_for_prediction
logger = logging.getLogger(__name__)
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

@router.post("/")
def predict_addiction_level(user_input: UserInput, request: Request):
    logger.info("Nhận được request dự đoán mới từ người dùng.")
    try:
        ml_components = request.app.state.ml_components

        input_dict = user_input.model_dump()
        df = pd.DataFrame([input_dict])
        
        X_scaled = preprocess_for_prediction(df, preprocessor_bundle=ml_components)
        
        model = ml_components["model"]
        prediction_encoded = model.predict(X_scaled)
        
        target_encoder = ml_components["target_encoder"]
        prediction_real = target_encoder.inverse_transform(prediction_encoded)
        
        result = int(prediction_real[0])
        percentile = ml_components["score_percentiles"].get(result, 0.0)
        logger.info(f"✅ Dự đoán hoàn tất. Mức độ nghiện: {result}")
        
        return {
            "status": "success",
            "predicted_addiction_score": result,
            "better_than_percentage": percentile
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình xử lý request dự đoán: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý nội bộ: {str(e)}")