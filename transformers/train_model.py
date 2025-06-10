import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from typing import Tuple
import mlflow

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def train_model(data: Tuple[pd.DataFrame, pd.DataFrame], *args, **kwargs) -> Tuple[LinearRegression, DictVectorizer]:
    """
    Vectorizes data, trains a LinearRegression model, and logs all steps,
    parameters, metrics, and artifacts (including model signature) to MLflow.
    """
    logger = kwargs.get('logger')
    df_train, df_val = data

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("nyc-taxi-experiment")

    logger.info("Starting MLflow run...")
    with mlflow.start_run():
        
        mlflow.log_param("model_class", "LinearRegression")
        mlflow.log_param("training_set_size", len(df_train))
        mlflow.log_param("validation_set_size", len(df_val))

        categorical = ['PULocationID', 'DOLocationID']
        train_dicts = df_train[categorical].astype(str).to_dict(orient='records')
        val_dicts = df_val[categorical].astype(str).to_dict(orient='records')

        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)
        X_val = dv.transform(val_dicts)
        
        mlflow.log_param("vectorizer_vocab_size", len(dv.vocabulary_))

        y_train = df_train['duration'].values
        y_val = df_val['duration'].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        val_r2_score = model.score(X_val, y_val)
        mlflow.log_metric("validation_r2_score", val_r2_score)
        logger.info(f"Validation R^2 score: {val_r2_score}")

        intercept = model.intercept_
        mlflow.log_metric("intercept", intercept)
        logger.info(f"Model intercept: {intercept}")
        
        # --- ADDED CODE TO CREATE AN INPUT EXAMPLE AND LOG WITH SIGNATURE ---
        # Take the first row of the training data as an example
        input_example = X_train[0]
        
        logger.info("Logging model with signature to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="linear_regression_model",
            input_example=input_example
        )
        # --------------------------------------------------------------------

    logger.info("✅ MLflow run completed successfully.")
    
    return model, dv