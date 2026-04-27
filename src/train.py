import joblib
import numpy as np

from preprocess import load_and_preprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load Data
data = load_and_preprocess("../data/train.tsv")

# 2. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))      #increased from 500 to 2000 and added n-grams
X = vectorizer.fit_transform(data["full_text"])

# 3. Label Encoding
encoder = LabelEncoder()
y = encoder.fit_transform(data["label"])

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# using GA
use_ga = True  # toggle this

if use_ga:
    selected_features = np.load("../artifacts/selected_features.npy")
    X_train = X_train[:, selected_features]
    X_test = X_test[:, selected_features]
    print(f"Using GA features: {len(selected_features)}")

# 5. Train Logistic Regression (Balanced)
model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print("\n=== LOGISTIC REGRESSION (BALANCED) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Save Artifacts
joblib.dump(model, "../models/logistic.pkl")
joblib.dump(vectorizer, "../artifacts/vectorizer.pkl")
joblib.dump(encoder, "../artifacts/encoder.pkl")

print("\nModel and artifacts saved successfully!")