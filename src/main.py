from data_loader import load_data
from text_preprocessing import clean_text, lemmatize_text
from model import create_model, train_model, predict
import joblib
def main():
    # Load Data
    train_data, valid_data = load_data('data/Train.csv', 'data/Valid.csv')
    
    # Preprocess Data
    train_data["clean_text"] = train_data["text"].apply(clean_text).apply(lemmatize_text)
    valid_data["clean_text"] = valid_data["text"].apply(clean_text).apply(lemmatize_text)
    
    X_train = train_data["clean_text"]
    y_train = train_data["label"]
    X_test = valid_data["clean_text"]
    y_test = valid_data["label"]

    # Create and Train Model
    model = create_model()
    train_model(model, X_train, y_train)


    # Save the trained model using joblib
    model_path = 'model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    # Example Input
    input_text = ["Such a bad movie"]
    predicted_class, predicted_proba = predict(model, input_text)
    
    label_map = {0: "Negative", 1: "Positive"}
    print("Predicted class:", label_map[predicted_class[0]])
    print("Class probabilities:", predicted_proba[0])

if __name__ == "__main__":
    main()
