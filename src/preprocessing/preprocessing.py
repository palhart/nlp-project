import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import tiktoken
import pandas as pd

def load_data(filename):
    df = pd.read_csv('../data/' + filename)
    return df

def preprocess_data(df, to_lower=True, remove_punct=True, remove_stops=True, lemmatize=True):
    nltk.download('stopwords')
    nltk.download('wordnet')

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

# Encoding choices: ["r50k_base", "p50k_base", "cl100k_base", "gpt-4"]
def tokenize_bpe(text, encoding_name="gpt-4"):
    if encoding_name == "gpt-4":
        encoding = tiktoken.encoding_for_model("gpt-4")
    else:
        encoding = tiktoken.get_encoding(encoding_name)

    tokens = encoding.encode(text)

    return tokens

def tokenize_nltk(text):
    nltk.download('punkt')

    sentences = sent_tokenize(text)
    sentences = [word_tokenize(s) for s in sentences]
    return sentences