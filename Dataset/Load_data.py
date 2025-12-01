import pandas as pd

df = pd.read_csv("Dataset/spam.csv", encoding="latin-1")  # common encoding for this dataset
print(df.head())
print(df.shape)
print(df.info())
print(df.columns)
print(df.isna().sum())
print(df["v1"].value_counts())
print(df.duplicated().sum())