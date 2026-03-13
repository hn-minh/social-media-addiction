import logging
import joblib
import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score

from src import config

logger = logging.getLogger(__name__)

def train_models(X_train, y_train, X_test, y_test):
    logger.info("Bắt đầu quá trình huấn luyện và chọn lọc model cùng MLflow...")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "SVC": SVC(kernel='linear', C=1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    }

    best_model_name = None
    best_model = None
    best_score = 0.0

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    train_dataset = mlflow.data.from_numpy(features=X_train, targets=y_train, name="SocialMedia_Train_Data")
    test_dataset = mlflow.data.from_numpy(features=X_test, targets=y_test, name="SocialMedia_Test_Data")

    for name, model in models.items():
        logger.info(f"Đang chạy và log model: {name}...")
        
        # Mở một "Run" mới trong MLflow cho mỗi thuật toán
        with mlflow.start_run(run_name=name):

            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(test_dataset, context="evaluation")
            
            # Huấn luyện
            model.fit(X_train, y_train)
            
            # Dự đoán
            y_pred = model.predict(X_test)
            
            # Chấm điểm
            score = f1_score(y_test, y_pred, average='weighted')
            acc = accuracy_score(y_test, y_pred)
            
            # --- SỨC MẠNH CỦA MLFLOW NẰM Ở ĐÂY ---
            # Tự động lưu toàn bộ tham số của model (n_estimators, max_depth, kernel...)
            mlflow.log_params(model.get_params())
            
            # Lưu điểm số để so sánh
            mlflow.log_metric("f1_weighted", score)
            mlflow.log_metric("accuracy", acc)
            
            # Lưu lại chính cấu trúc model đó lên MLflow
            mlflow.sklearn.log_model(model, artifact_path=name)
            
            logger.info(f"[{name}] - F1-Score: {score:.4f} | Accuracy: {acc:.4f}")

            # Cập nhật model tốt nhất trong phiên chạy này
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name

    logger.info("="*50)
    logger.info(f"🏆 MODEL CHIẾN THẮNG: {best_model_name} với F1-Score: {best_score:.4f}")
    logger.info("="*50)

    return best_model, best_model_name, best_score

logger = logging.getLogger(__name__)

def save_artifacts(best_model, best_model_name, best_score, scaler, encoders_dict, target_encoder, score_percentiles):
    logger.info("Bắt đầu đóng gói và đánh giá Model trên Registry...")

    preprocessor_bundle = {
        "scaler": scaler,
        "encoders": encoders_dict,
        "target_encoder": target_encoder,
        "score_percentiles": score_percentiles
    }
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temp_preprocessor_path = config.MODEL_DIR / "preprocessor_bundle.pkl"
    joblib.dump(preprocessor_bundle, temp_preprocessor_path)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    model_name = "SocialMediaAddiction_Classifier"
    alias_name = "production"

    # 2. Đẩy Model phiên bản mới lên MLflow
    with mlflow.start_run(run_name="Model_Evaluation_and_Registration") as run:
        
        # log_model trả về object chứa thông tin version vừa đăng ký
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        mlflow.log_artifact(local_path=str(temp_preprocessor_path), artifact_path="preprocessors")
        mlflow.log_param("winning_algorithm", best_model_name)
        # Ghi lại điểm F1 của phiên bản này để lần sau có cái so sánh
        mlflow.log_metric("validation_f1_score", best_score) 
        
        new_version = model_info.registered_model_version
        logger.info(f"Đã đăng ký Model mới thành công: Version {new_version}")

    # 3. LOGIC GÁN NHÃN (MODEL PROMOTION) BẰNG MLFLOW CLIENT
    client = MlflowClient()

    try:
        # Tìm xem hệ thống đã có model nào mang nhãn 'production' chưa
        prod_model = client.get_model_version_by_alias(name=model_name, alias=alias_name)
        
        # Lấy lại điểm F1 của model production hiện tại từ Run ID của nó
        prod_run = client.get_run(prod_model.run_id)
        prod_score = prod_run.data.metrics.get("validation_f1_score", 0.0)
        
        logger.info(f"Model Production hiện tại: Version {prod_model.version} (F1: {prod_score:.4f})")
        logger.info(f"Model mới huấn luyện: Version {new_version} (F1: {best_score:.4f})")

        # So sánh: Nếu model mới xịn hơn, tước ngôi production của model cũ và gán cho model mới
        if best_score > prod_score:
            logger.info("🎉 Thử thách THÀNH CÔNG! Model mới hoạt động tốt hơn.")
            logger.info(f"Đang thăng cấp Version {new_version} lên nhãn '{alias_name}'...")
            client.set_registered_model_alias(name=model_name, alias=alias_name, version=new_version)
        else:
            logger.warning("❌ Thử thách THẤT BẠI! Model mới không tốt bằng model hiện tại.")
            logger.info(f"Giữ nguyên nhãn '{alias_name}' cho Version {prod_model.version}.")

    except MlflowException:
        # Nếu exception xảy ra, nghĩa là chưa từng có model nào mang nhãn production (chạy lần đầu)
        logger.info(f"Chưa có model '{alias_name}' nào trên hệ thống. Đang gán nhãn cho Version {new_version}...")
        client.set_registered_model_alias(name=model_name, alias=alias_name, version=new_version)

    logger.info("Hoàn tất quá trình lưu trữ và gán nhãn.")