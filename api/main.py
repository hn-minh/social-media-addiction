import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import joblib
import os
from dotenv import load_dotenv

from api.routers import predict, collect

logger = logging.getLogger(__name__)
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Đang khởi động API Server... Kết nối MLflow để tải Model @production.")
    
    app.state.ml_components = {}
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        model_name = "SocialMediaAddiction_Classifier"
        alias_name = "production"

        prod_model = client.get_model_version_by_alias(name=model_name, alias=alias_name)
        run_id = prod_model.run_id
        
        logger.info(f"Đã tìm thấy Model Production: Version {prod_model.version} (Run ID: {run_id})")

        model_uri = f"models:/{model_name}@{alias_name}"
        app.state.ml_components["model"] = mlflow.pyfunc.load_model(model_uri)
        
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, 
            artifact_path="preprocessors/preprocessor_bundle.pkl"
        )
        preprocessor_bundle = joblib.load(artifact_path)
        
        # Nạp vào RAM
        app.state.ml_components["scaler"] = preprocessor_bundle["scaler"]
        app.state.ml_components["encoders"] = preprocessor_bundle["encoders"]
        app.state.ml_components["target_encoder"] = preprocessor_bundle["target_encoder"]
        app.state.ml_components["score_percentiles"] = preprocessor_bundle["score_percentiles"]
        
        logger.info("✅ Đã tải thành công Model và các công cụ Preprocessing vào bộ nhớ.")
        yield
        
    except MlflowException:
        logger.critical(f"LỖI: Không tìm thấy model nào mang nhãn '@{alias_name}' trên MLflow.")
        raise
    except Exception as e:
        logger.critical(f"Lỗi hệ thống khi khởi động API: {str(e)}")
        raise e
    finally:
        logger.info("Đang tắt Server và giải phóng bộ nhớ.")
        app.state.ml_components.clear()

# Khởi tạo App
app = FastAPI(
    title="Social Media Addiction API", 
    description="API dự đoán mức độ nghiện mạng xã hội dựa trên hành vi người dùng.",
    version="1.0.0",
    lifespan=lifespan
)

# KẾT NỐI ROUTERS
app.include_router(predict.router)
app.include_router(collect.router)

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    # Lưu ý: Lệnh chạy giờ phải trỏ đúng đường dẫn module 'api.main:app'
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)