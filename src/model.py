from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def create_model():
    return make_pipeline(CountVectorizer(), MultinomialNB())

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def predict(model, input_text):
    return model.predict(input_text), model.predict_proba(input_text)

