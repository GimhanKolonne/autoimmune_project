import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

from . import config


def load_artifacts(prefer: str = "ensemble") -> Dict[str, Any]:
    """Load saved model, label encoder, and feature list. Falls back if preferred model is missing."""
    try:
        label_encoder_path = config.MODELS_DIR / "label_encoder.joblib"
        features_path = config.RESULTS_DIR / "selected_features.json"
        
        if not label_encoder_path.exists():
            raise ValueError(f"Label encoder not found: {label_encoder_path}")
        if not features_path.exists():
            raise ValueError(f"Selected features not found: {features_path}")
            
        label_encoder = joblib.load(label_encoder_path)
        
        with open(features_path, 'r') as f:
            selected_features = json.load(f)
        
        # Load preferred model
        ensemble_path = config.MODELS_DIR / "model_Ensemble.joblib"
        xgboost_path = config.MODELS_DIR / "model_XGBoost.joblib"
        
        model = None
        model_name = None
        
        if prefer == "ensemble":
            if ensemble_path.exists():
                model = joblib.load(ensemble_path)
                model_name = "Ensemble"
            elif xgboost_path.exists():
                model = joblib.load(xgboost_path)
                model_name = "XGBoost"
                print("Warning: Ensemble not found, using XGBoost")
            else:
                raise ValueError("Neither Ensemble nor XGBoost model found")
                
        elif prefer == "xgboost":
            if xgboost_path.exists():
                model = joblib.load(xgboost_path)
                model_name = "XGBoost"
            elif ensemble_path.exists():
                model = joblib.load(ensemble_path)
                model_name = "Ensemble"
                print("Warning: XGBoost not found, using Ensemble")
            else:
                raise ValueError("Neither XGBoost nor Ensemble model found")
        else:
            raise ValueError("prefer must be 'ensemble' or 'xgboost'")
        
        return {
            "model": model,
            "label_encoder": label_encoder,
            "selected_features": selected_features,
            "model_name": model_name
        }
        
    except Exception as e:
        raise ValueError(f"Error loading artifacts: {str(e)}")


def validate_patient_input(df: pd.DataFrame, selected_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Validate and clean patient input: check columns, coerce types, clip negatives. Returns (cleaned_df, warnings)."""
    warnings = []
    clean_df = df.copy()
    
    missing_features = [f for f in selected_features if f not in clean_df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    extra_columns = [col for col in clean_df.columns if col not in selected_features]
    if extra_columns:
        warnings.append(f"Dropping extra columns: {extra_columns}")
        clean_df = clean_df.drop(columns=extra_columns)
    
    # Reorder to match training feature order
    clean_df = clean_df[selected_features]
    original_dtypes = clean_df.dtypes
    clean_df = clean_df.apply(pd.to_numeric, errors='coerce')
    
    for col in clean_df.columns:
        if clean_df[col].isna().any() and not original_dtypes[col] in ['float64', 'int64']:
            warnings.append(f"Non-numeric values in {col} converted to NaN")
    
    if clean_df.isna().any().any():
        warnings.append("NaN values found and replaced with 0")
        clean_df = clean_df.fillna(0)
    
    negative_cols = []
    for col in clean_df.columns:
        if (clean_df[col] < 0).any():
            negative_cols.append(col)
            clean_df[col] = clean_df[col].clip(lower=0)
    
    if negative_cols:
        warnings.append(f"Negative values clipped to 0 in columns: {negative_cols}")
    
    return clean_df, warnings


def preprocess_like_training(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the training preprocessing: numeric coerce, pseudocount, relative abundance, log1p."""
    processed_df = df.copy()
    
    # Defensive re-clean (validation should have handled this already)
    processed_df = processed_df.apply(pd.to_numeric, errors='coerce')
    processed_df = processed_df.fillna(0)
    processed_df = processed_df.clip(lower=0)
    
    # Pseudocount to avoid log(0)
    processed_df = processed_df + 1e-9
    
    # Relative abundance (row-wise normalisation)
    row_sums = processed_df.sum(axis=1).replace(0, 1e-9)
    processed_df = processed_df.div(row_sums, axis=0)
    
    processed_df = np.log1p(processed_df)
    
    return processed_df


def predict_patient(df_raw: pd.DataFrame, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Validate, preprocess, and predict for a single patient. Returns label, probabilities, confidence."""
    try:
        model = artifacts["model"]
        label_encoder = artifacts["label_encoder"]
        selected_features = artifacts["selected_features"]
        
        df_clean, warnings = validate_patient_input(df_raw, selected_features)
        
        for warning in warnings:
            print(f"Warning: {warning}")
        
        df_processed = preprocess_like_training(df_clean)
        
        prediction_index = model.predict(df_processed)[0]
        predicted_label = label_encoder.inverse_transform([prediction_index])[0]
        
        probabilities = {}
        confidence = 1.0
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df_processed)[0]
            class_names = label_encoder.classes_
            probabilities = {class_name: float(prob) for class_name, prob in zip(class_names, proba)}
            confidence = float(np.max(proba))
        else:
            # Fallback: hard assignment when predict_proba unavailable
            class_names = label_encoder.classes_
            probabilities = {class_name: 0.0 for class_name in class_names}
            probabilities[predicted_label] = 1.0
        
        return {
            "predicted_label": predicted_label,
            "probabilities": probabilities,
            "predicted_index": int(prediction_index),
            "confidence": confidence
        }
        
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")


def clinical_decision_support(patient_data: Dict[str, float], prefer_model: str = "ensemble") -> Dict[str, Any]:
    """End-to-end inference pipeline: load model, predict, attach clinical certainty level."""
    try:
        artifacts = load_artifacts(prefer=prefer_model)
        df_patient = pd.DataFrame([patient_data])
        result = predict_patient(df_patient, artifacts)
        
        confidence = result["confidence"]
        if confidence >= 0.8:
            certainty = "High"
        elif confidence >= 0.6:
            certainty = "Moderate"
        else:
            certainty = "Low"
        
        result["clinical_certainty"] = certainty
        result["model_used"] = artifacts["model_name"]
        result["recommendation"] = f"Predicted condition: {result['predicted_label']} (Confidence: {certainty})"
        
        return result
        
    except Exception as e:
        raise ValueError(f"Clinical decision support error: {str(e)}")


def batch_predict_patients(df_patients: pd.DataFrame, prefer_model: str = "ensemble") -> pd.DataFrame:
    """Predict for multiple patients at once. Returns DataFrame with labels, probabilities, confidence."""
    try:
        artifacts = load_artifacts(prefer=prefer_model)
        
        df_clean, warnings = validate_patient_input(df_patients, artifacts["selected_features"])
        df_processed = preprocess_like_training(df_clean)
        
        model = artifacts["model"]
        label_encoder = artifacts["label_encoder"]
        
        predictions = model.predict(df_processed)
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        results = pd.DataFrame({
            "patient_index": range(len(df_patients)),
            "predicted_label": predicted_labels,
            "predicted_index": predictions
        })
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df_processed)
            class_names = label_encoder.classes_
            
            for i, class_name in enumerate(class_names):
                results[f"prob_{class_name}"] = probabilities[:, i]
            
            results["confidence"] = np.max(probabilities, axis=1)
        
        return results
        
    except Exception as e:
        raise ValueError(f"Batch prediction error: {str(e)}")