import nltk
nltk.download('punkt')

import numpy as np  
import pandas as pd
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Load and Inspect the Dataset
# ---------------------------
df = pd.read_csv("spotify_millsongdata.csv")
print("Dataset Head:")
print(df.head(5))
print("Dataset Tail:")
print(df.tail(5))
print("Dataset Shape:", df.shape)
print("Null Values in Dataset:")
print(df.isnull().sum())

# ---------------------------
# Sample the Data and Clean Up
# ---------------------------
# Sample 5000 rows with a fixed random state and drop the 'link' column
df = df.sample(5000, random_state=42).drop('link', axis=1).reset_index(drop=True)
print("\nSampled DataFrame Head:")
print(df.head(10))
print("\nFirst element in 'text' column:")
print(df['text'][0])
print("DataFrame shape after sampling:", df.shape)

# Preprocess the text: convert to lowercase, remove extra spaces and newlines
df['text'] = df['text'].str.lower() \
                      .replace(r'\s+', ' ', regex=True) \
                      .replace(r'\n', ' ', regex=True)
print("\nDataFrame tail after text preprocessing:")
print(df.tail(5))

# ---------------------------
# Text Preprocessing: Stemming
# ---------------------------
stemmer = PorterStemmer()

def token(txt):
    tokens = nltk.word_tokenize(txt)
    stemmed = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)

# Test the token function
print("\nToken function test for 'you are good':")
print(token("you are good"))

# ---------------------------
# TF-IDF Vectorization and Similarity Computation
# ---------------------------
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidf.fit_transform(df['text'])
print("\nTF-IDF Matrix shape:", matrix.shape)

# Compute cosine similarity based on the TF-IDF matrix
similar = cosine_similarity(matrix)
print("\nCosine similarity for first document (sample):")
print(similar[0])

# ---------------------------
# Check for a Specific Song (Example: "What You Need")
# ---------------------------
filtered = df[df['song'].str.strip().str.lower() == 'what you need'.lower()]
if filtered.empty:
    print("\nSong 'What You Need' not found in the DataFrame.")
else:
    index = filtered.index[0]
    print("\nIndex of 'What You Need':", index)
    print(filtered)

# ---------------------------
# Updated Music Recommender Function
# ---------------------------
def recommender(song_name, top_n=20):
    """
    Given a song name, returns the top_n similar songs based on cosine similarity.
    """
    # Clean the input for robust matching
    song_name_clean = song_name.strip().lower()
    idx_filter = df['song'].str.strip().str.lower() == song_name_clean
    filtered = df[idx_filter]
    
    if filtered.empty:
        return f"Song '{song_name}' not found in the DataFrame."
    
    # Get the index of the first matching song
    idx = filtered.index[0]
    
    # Compute cosine similarity distances and sort them (highest first)
    distances = sorted(list(enumerate(similar[idx])), reverse=True, key=lambda x: x[1])
    
    # Collect recommendations, skipping the first one (itself)
    recommendations = []
    for i in distances[1:top_n+1]:
        recommendations.append(df.iloc[i[0]].song)
    
    return recommendations

# ---------------------------
# Test the Recommender Function
# ---------------------------
print("\nRecommendations for 'Crazy':")
print(recommender("Crazy"))

# ---------------------------
# Save Processed Data: Pickle Dump
# ---------------------------
pickle.dump(similar, open('similar.pkl', 'wb'))
pickle.dump(df, open('df.pkl', 'wb'))
print("\nPickle dump completed: 'similar.pkl' and 'df.pkl' have been created.")
