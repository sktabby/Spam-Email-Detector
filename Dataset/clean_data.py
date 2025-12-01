import pandas as pd
df = pd.read_csv("Dataset/spam.csv", encoding="latin-1")
df = df.drop(columns=['Unnamed: 2', "Unnamed: 3", "Unnamed: 4"])
df = df.rename(columns={'v1':'label','v2':'text'})
df = df.drop_duplicates()


print(df.head())
print(df.shape)
print(df.duplicated().sum())
print(df['label'].value_counts())