import pandas as pd
from src.config import DATA_PROCESSED_PATH
from src.preprocessing import preprocess_for_clustering

df = pd.read_csv(DATA_PROCESSED_PATH + "customers_features.csv")
X = preprocess_for_clustering(df)
print(X.shape)
