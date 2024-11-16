import re
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import nltk

nltk.download("popular")

# Preprocessing Functions
stop_words = stopwords.words("english")

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def remove_hashtag(text):
    return re.sub(r"#\w+", "", text)

def remove_punctuation(text):
    return re.sub(r"[!\"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~،؟…«“”:”]", "", text)

def remove_mention(text):
    return re.sub(r"@\w+", "", text)

def remove_urls(text):
    return re.sub(r"http\S+|www\S+", "", text)

def remove_numbers(text):
    return re.sub(r"\d+", "", text)

def remove_multiple_whitespace(text):
    return re.sub(r"\s{2,}", " ", text)

def clean_text(text):
    text = text.lower()
    text = remove_urls(text)
    text = remove_hashtag(text)
    text = remove_mention(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stop_words(text)
    text = remove_multiple_whitespace(text)
    return text.strip()

# Stemming and Lemmatization
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

def stem_text(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
