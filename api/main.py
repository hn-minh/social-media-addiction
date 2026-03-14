import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    logger.info("Starting API server. Fetching @production model from MLflow.")
    
    app.state.ml_components = {}
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        model_name = "SocialMediaAddiction_Classifier"
        alias_name = "production"

        prod_model = client.get_model_version_by_alias(name=model_name, alias=alias_name)
        run_id = prod_model.run_id
        
        logger.info(f"Production model found: Version {prod_model.version} (Run ID: {run_id})")

        model_uri = f"models:/{model_name}@{alias_name}"
        app.state.ml_components["model"] = mlflow.pyfunc.load_model(model_uri)
        
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, 
            artifact_path="preprocessors/preprocessor_bundle.pkl"
        )
        preprocessor_bundle = joblib.load(artifact_path)
        
        app.state.ml_components["scaler"] = preprocessor_bundle["scaler"]
        app.state.ml_components["encoders"] = preprocessor_bundle["encoders"]
        app.state.ml_components["target_encoder"] = preprocessor_bundle["target_encoder"]
        app.state.ml_components["score_percentiles"] = preprocessor_bundle["score_percentiles"]
        
        logger.info("Model and preprocessors successfully loaded into memory.")
        yield
        
    except MlflowException:
        logger.critical(f"Error: Model alias '@{alias_name}' not found in MLflow.")
        raise
    except Exception as e:
        logger.critical(f"API startup failed: {str(e)}")
        raise e
    finally:
        logger.info("Shutting down server and releasing memory.")
        app.state.ml_components.clear()

app = FastAPI(
    title="Social Media Addiction API", 
    description="API to predict social media addiction levels based on user behavior.",
    version="1.0.0",
    lifespan=lifespan
)

allowed_frontend_url = os.getenv("FRONTEND_URL")
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    allowed_frontend_url
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(collect.router)

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)