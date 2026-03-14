import logging
import sys
from data_ingestion import load_data
from preprocessing import engineer_features, preprocess_and_split
from train import train_models, save_artifacts

logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("Starting training pipeline.")

    try:
        logger.info("STEP 1: LOAD DATASET.")
        df_raw = load_data()

        logger.info("STEP 2: PREPROCESSING DATA.")
        df_engineered = engineer_features(df_raw)
        X_train, X_test, y_train, y_test, scaler, encoders_dict, target_encoder, score_percentiles = preprocess_and_split(df_engineered)

        logger.info("STEP 3: TRAINING AND EVALUATION")
        best_model, best_model_name, best_score = train_models(
            X_train, y_train, X_test, y_test
        )

        save_artifacts(best_model, best_model_name, best_score, scaler, encoders_dict, target_encoder, score_percentiles)

        logger.info("PIPELINE COMPLETED SUCCESSFULLY.")
        
    except Exception as e:
        logger.critical(f"PIPELINE FAILED: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()