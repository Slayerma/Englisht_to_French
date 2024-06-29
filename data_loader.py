import collections
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(path):
    with open(path, "r") as f:
        return f.read().split('\n')

def tokenize(sentences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    return tokenizer.texts_to_sequences(sentences), tokenizer

def pad_sequences_with_max_length(sequences, max_length=None):
    if (max_length is None):
        max_length = max(len(sentence) for sentence in sequences)
    return pad_sequences(sequences, maxlen=max_length, padding='post')

def preprocess(english_sentences, french_sentences):
    preproc_english, english_tokenizer = tokenize(english_sentences)
    preproc_french, french_tokenizer = tokenize(french_sentences)
    
    preproc_english = pad_sequences_with_max_length(preproc_english)
    preproc_french = pad_sequences_with_max_length(preproc_french)
    
    preproc_french = preproc_french.reshape(*preproc_french.shape, 1)
    
    return preproc_english, preproc_french, english_tokenizer, french_tokenizer
