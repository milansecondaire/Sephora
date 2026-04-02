import pandas as pd
from src.preprocessing import preprocess_for_clustering

def test_return_transformers():
    df_mini = pd.DataFrame({
        "recency_days": [10.0, 20.0, 30.0],
        "age": [20.0, 30.0, 40.0],
        "customer_city": ["Paris", "Lyon", "Paris"],
        "gender": ["Men", "Women", "Men"]
    })
    
    # Par défaut, juste le dataframe
    res = preprocess_for_clustering(df_mini)
    assert isinstance(res, pd.DataFrame)
    
    # Avec flag, renvoie tuple
    res_df, transformers = preprocess_for_clustering(df_mini, return_transformers=True)
    assert isinstance(res_df, pd.DataFrame)
    assert "scaler" in transformers
    assert transformers["scaler"].with_mean
    assert "customer_city" in transformers["frequency_encoding"]
