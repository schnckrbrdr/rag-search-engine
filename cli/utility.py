import os
import json
import string
from nltk.stem import PorterStemmer

# load movie data and stopwords
def load_movies() -> list[dict]:
    try:
        movies_file = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "movies.json"))
        movies_json = json.load(movies_file)        
        return movies_json
    except Exception as e:
        print(e)
        return None

def load_stopwords() -> list[str]:
    try:
        stop_words_file = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stopwords.txt"))
        stopwords = stop_words_file.read().splitlines()
        return stopwords
    except Exception as e:
        print(e)
        return None

# Tokenize passed text and remove tokens listed in stopwords list
def tokenize(text: str, stopwords: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    text_tokens = strip_punctuation(text).split()
    final_tokens = []
    for text_token in text_tokens:
        if text_token not in stopwords:
            final_tokens.append(stemmer.stem(text_token))
    return final_tokens

def strip_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation)).lower()