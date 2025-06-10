import pandas as pd
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def load_data(*args, **kwargs) -> pd.DataFrame:
    """
    Loads the complete NYC taxi dataset from a parquet file into one DataFrame.
    """
    logger = kwargs.get('logger')
    data_file = "/home/src/data/yellow_tripdata_2023-03.parquet"
    logger.info(f"Loading full dataset from {data_file}...")
    
    df = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(df)} rows.")
    
    return df