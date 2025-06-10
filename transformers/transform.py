import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(df: pd.DataFrame, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Performs feature engineering and cleaning.
    2. Splits the data into training and validation sets.
    """
    logger = kwargs.get('logger')

    # Feature Engineering
    logger.info("Performing feature engineering...")
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df.duration.dt.total_seconds() / 60
    
    # Filtering
    logger.info(f"Original row count: {len(df)}")
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    logger.info(f"Row count after filtering: {len(df)}")

    # Splitting data
    logger.info("Splitting data into training and validation sets...")
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Training set size: {len(df_train)}")
    logger.info(f"Validation set size: {len(df_val)}")

    return df_train, df_val