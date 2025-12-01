import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords once
nltk.download('stopwords')

# 1️⃣ Load raw dataset
df = pd.read_csv("Dataset/spam.csv", encoding="latin-1")  # or your original file name
print("Original columns:", df.columns)

# 2️⃣ Keep only useful columns and rename
df = df[['v1', 'v2']]                      # drop Unnamed: 2,3,4 directly
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
print("After renaming:", df.columns)

# 3️⃣ Setup stopwords and stemmer
stop = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stop]
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)

# 4️⃣ Apply preprocessing
df['text'] = df['text'].apply(preprocess)

# 5️⃣ Drop duplicates again just to be safe
df = df.drop_duplicates()

print(df.head(10))
print(df.shape)

# 6️⃣ Save final cleaned + preprocessed dataset
df.to_csv("preprocessed_spam.csv", index=False)
print("✅ Saved to preprocessed_spam.csv")
