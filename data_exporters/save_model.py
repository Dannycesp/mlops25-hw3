from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction import DictVectorizer
import joblib
from pathlib import Path
from typing import Tuple

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def save_model(data: Tuple[SGDRegressor, DictVectorizer], *args, **kwargs) -> None:
    """
    Saves the trained model and the fitted vectorizer to files,
    and logs the size of the model file.
    """
    logger = kwargs.get('logger')
    model, dv = data  # Unpack the tuple from the previous block

    output_dir = Path("/home/src/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "sgd_model.joblib"
    dv_path = output_dir / "dv.joblib"

    logger.info(f"📦 Saving model to: {model_path}")
    joblib.dump(model, model_path)
    
    # --- ADDED CODE TO LOG FILE SIZE ---
    model_size_bytes = model_path.stat().st_size
    logger.info(f"MODEL_SIZE_BYTES: {model_size_bytes}")
    # ------------------------------------
    
    logger.info(f"📦 Saving vectorizer to: {dv_path}")
    joblib.dump(dv, dv_path)

    logger.info("✅ All artifacts saved successfully.")