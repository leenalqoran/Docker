from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that uses an NLP model to predict the sentiment of movie reviews.",
    version="0.1",
)

# Load the trained model
model_path = "model.pkl"  # Path to your saved model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {model_path}. Ensure the model is saved in the correct path.")

# Define input model for validation
class ReviewInput(BaseModel):
    review: str  # The input is expected to be a single string of review text

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Model API. Use the /predict-review endpoint to analyze sentiment."}

# Define the endpoint for sentiment prediction
@app.post("/predict-review")
def predict_sentiment(input_data: ReviewInput):
    review = input_data.review
    try:
        # Predict sentiment for the input text
        predicted_class = model.predict([review])
        predicted_proba = model.predict_proba([review])

        # Map the prediction to a human-readable label
        label_map = {0: "Negative", 1: "Positive"}
        result = {
            "prediction": label_map[predicted_class[0]],
            "confidence": max(predicted_proba[0])  # Probability of the predicted class
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
