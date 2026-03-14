import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src import config

logger = logging.getLogger(__name__)

def engineer_features(df):
    logger.info("Starting feature engineering.")
    df = df.copy()

    if "Student_ID" in df.columns:
        df.drop("Student_ID", axis=1, inplace=True)
        
    if "Addicted_Score" in df.columns:
        class_counts = df['Addicted_Score'].value_counts()
        valid_classes = class_counts[class_counts >= config.MIN_SAMPLES_REQUIRED].index
        dropped_classes = class_counts[class_counts < config.MIN_SAMPLES_REQUIRED].index
        
        if len(dropped_classes) > 0:
            logger.warning(f"Dropped classes with <{config.MIN_SAMPLES_REQUIRED} samples: {list(dropped_classes)}")
            df = df[df['Addicted_Score'].isin(valid_classes)].reset_index(drop=True)
        else:
            logger.info("All classes meet minimum sample requirements. No classes dropped.")

    df['Usage_Category'] = pd.cut(
        df['Avg_Daily_Usage_Hours'], 
        bins=[0, 2, 4, 6, float('inf')], 
        labels=['Low', 'Moderate', 'High', 'Very High']
    )

    df['Sleep_Quality'] = df['Sleep_Hours_Per_Night'].apply(
        lambda x: 'Poor' if x < 6 else 'Good' if x >= 7 else 'Fair'
    )

    df['Mental_Health_Category'] = pd.cut(
        df['Mental_Health_Score'], 
        bins=[0, 4, 7, 10], 
        labels=['Poor', 'Moderate', 'Good']
    )

    df['Region'] = df['Country'].apply(
        lambda x: 'Asia' if x in config.ASIAN_COUNTRIES 
        else 'Europe' if x in config.EUROPEAN_COUNTRIES 
        else 'Americas' if x in config.AMERICAN_COUNTRIES 
        else 'Other'
    )

    df['Wellness_Score'] = (df['Mental_Health_Score'] + (df['Sleep_Hours_Per_Night'] / 9 * 10)) / 2

    cols_to_drop = ['Academic_Level', 'Gender', 'Sleep_Quality']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    logger.info("Feature engineering completed.")
    return df

def preprocess_and_split(df):
    logger.info("Encoding data.")
    df = df.copy()

    X = df.drop(columns=["Addicted_Score"])
    y_raw = df["Addicted_Score"]
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y_raw)

    score_percentiles = {}
    counts = y_raw.value_counts(normalize=True) 
    
    for score in counts.index:
        worse_proportion = counts[counts.index > score].sum()
        score_percentiles[score] = round(worse_proportion * 100, 2)

    object_cols = X.select_dtypes(include=['object', 'category']).columns
    encoders_dict = {}
    
    for col in object_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders_dict[col] = le
    logger.info(f"Encoded {len(object_cols)} categorical columns.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=config.TEST_SIZE, random_state=537, stratify=y_encoded
    )
    logger.info("Data split successfully.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders_dict, target_encoder, score_percentiles


def preprocess_for_prediction(df_input, preprocessor_bundle):
    logger.info("Starting preprocessing for prediction API.")
    df = df_input.copy()

    df = engineer_features(df)

    encoders = preprocessor_bundle.get("encoders", {})
    scaler = preprocessor_bundle.get("scaler")

    object_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in object_cols:
        if col in encoders:
            le = encoders[col]
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                logger.warning(f"Unknown label detected in column '{col}'. Defaulting to -1.")
                df[col] = -1 
        else:
            logger.warning(f"Column '{col}' lacks a corresponding encoder. It will be ignored.")

    if hasattr(scaler, "feature_names_in_"):
        expected_cols = scaler.feature_names_in_
        df = df[expected_cols]

    X_scaled = scaler.transform(df)

    logger.info("Preprocessing for prediction completed.")
    return X_scaled