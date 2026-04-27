import joblib
import numpy as np

from preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split

# Load model and artifacts
model = joblib.load("../models/logistic.pkl")
vectorizer = joblib.load("../artifacts/vectorizer.pkl")
encoder = joblib.load("../artifacts/encoder.pkl")


# Fuzzy Logic Function
def fuzzy_op(label, prob):
    if prob > 0.65:
        if label == "Fake":
            return "Highly Fake"
        elif label == "Real":
            return "Highly Real"
        else:
            return "Leaning Uncertain"

    elif prob > 0.50:
        if label == "Fake":
            return "Mostly Fake"
        elif label == "Real":
            return "Mostly Real"
        else:
            return "Uncertain"

    else:
        return "Uncertain"


print("\n===== Testing on Dataset =====")

# Load dataset
data = load_and_preprocess("../data/train.tsv")

# Vectorize
X = vectorizer.transform(data["full_text"])
y = encoder.transform(data["label"])

# Split (same as training)
_, X_test, _, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Show first 10 predictions
for i in range(10):
    actual = encoder.inverse_transform([y_test[i]])[0]
    predicted = encoder.inverse_transform([y_pred[i]])[0]
    prob = np.max(y_prob[i])

    print(f"\nSample {i+1}")
    print("Actual:", actual)
    print("Predicted:", predicted)
    print("Confidence:", round(prob, 3))
    print("Fuzzy:", fuzzy_op(predicted, prob))

    # ===========================
# USER INPUT SYSTEM
# ===========================
print("\n===== Fake News Detection System =====")

while True:
    user_input = input("\nEnter news (or type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    input_vector = vectorizer.transform([user_input])

    pred = model.predict(input_vector)[0]
    label = encoder.inverse_transform([pred])[0]

    probs = model.predict_proba(input_vector)[0]
    max_prob = np.max(probs)

    print("\n--- RESULT ---")
    print("Prediction :", label)
    print("Confidence :", round(max_prob, 3))
    print("Fuzzy      :", fuzzy_op(label, max_prob))

    print("\n--- CLASS PROBABILITIES ---")
    for i, p in enumerate(probs):
        class_name = encoder.inverse_transform([i])[0]
        print(f"{class_name:10s}: {round(p, 3)}")