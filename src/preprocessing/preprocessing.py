import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import tiktoken
import pandas as pd
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def load_data(filename):
    df = pd.read_csv('../data/' + filename)
    return df

def preprocess_data(df, to_lower=True, remove_punct=True, remove_stops=True, lemmatize=True):
    def preprocess_text(text):
        preprocessed_text = text
        if to_lower:
            preprocessed_text = preprocessed_text.lower()
        if remove_punct:
            preprocessed_text = ''.join(char for char in preprocessed_text if char not in string.punctuation)
        if remove_stops:
            stop_words = set(stopwords.words('english'))
            preprocessed_text = ' '.join(token for token in preprocessed_text.split() if token not in stop_words)
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            preprocessed_text = ' '.join(lemmatizer.lemmatize(token) for token in preprocessed_text.split())

        return preprocessed_text
    df['lyrics'] = df['lyrics'].apply(preprocess_text)
    return df

def adjust_genre_distribution(df):
    def map_genre_to_category(genre):
        if genre in ['Hip-Hop', 'R&B', 'Electronic']:
            return 'R&B/Hip-Hop/Electronic'
        elif genre in ['Country', 'Folk']:
            return 'Country/Folk'
        else:
            return genre
        
    df['genre'] = df['genre'].apply(map_genre_to_category)
    df = df[~(df['genre'] == 'Other')]
    df = df[~(df['genre'] == 'Indie')]

    rocks_indices = df[df['genre'] == 'Rock'].index

    num_rocks_to_keep = 35000
    num_rocks_current = len(rocks_indices)


    if num_rocks_current > num_rocks_to_keep:
        keep_indices = np.random.choice(rocks_indices, num_rocks_to_keep, replace=False)
    else:
        keep_indices = rocks_indices

    df = df.drop(index=set(rocks_indices) - set(keep_indices))
    return df


# Encoding choices: ["r50k_base", "p50k_base", "cl100k_base", "gpt-4"]
def tokenize_bpe(text, encoding_name="gpt-4"):
    if encoding_name == "gpt-4":
        encoding = tiktoken.encoding_for_model("gpt-4")
    else:
        encoding = tiktoken.get_encoding(encoding_name)

    tokens = encoding.encode(text)

    return tokens

def tokenize_nltk(text):

    sentences = sent_tokenize(text)
    sentences = [word_tokenize(s) for s in sentences]
    return sentences