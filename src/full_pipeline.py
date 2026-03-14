import logging
import sys
from data_ingestion import load_data
from preprocessing import engineer_features, preprocess_and_split
from train import train_models, save_artifacts

logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("START TRAINING PIPELINE")

    try:
        df_raw = load_data()

        df_engineered = engineer_features(df_raw)
        X_train, X_test, y_train, y_test, scaler, encoders_dict, target_encoder, score_percentiles = preprocess_and_split(df_engineered)

        logger.info("--- BƯỚC 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ ---")
        best_model, best_model_name, best_score = train_models(
            X_train, y_train, X_test, y_test
        )

        save_artifacts(best_model, best_model_name, best_score, scaler, encoders_dict, target_encoder, score_percentiles)

        logger.info("========== PIPELINE HOÀN TẤT THÀNH CÔNG ==========")
        
    except Exception as e:
        logger.critical(f"PIPELINE FAILED. Detail: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()