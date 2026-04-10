import pandas as pd
import numpy as np

# Column names for LIAR dataset
columns = ["id", "label", "statement", "subject", "speaker","speaker_job", "state", "party","barely_true", "false", "half_true", "mostly_true", "pants_on_fire", "context"]

# Load dataset
data = pd.read_csv("data/train.tsv", sep="\t", header=None, names=columns)

# Show first rows
print(data.head())

# Fill missing values to avoid errors
text_columns = ["statement", "subject", "speaker","speaker_job", "state", "party", "context"]
for col in text_columns:
    data[col] = data[col].fillna("")    # filled the missing values

# Combine all important text fields into one
data["full_text"] = (data["statement"] + " " + data["subject"] + " " + data["speaker"] + " " +
                     data["speaker_job"] + " " + data["state"] + " " + data["party"] + " " + data["context"] )

# Keep only useful columns
data = data[["label", "full_text"]]

# Check output
print(data.head())

# Using TF-IDF & Encoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Convert text to numerical features
vectorizer = TfidfVectorizer(max_features=500)      # changing 5000 to 500 features

X = vectorizer.fit_transform(data["full_text"]).toarray()

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(data["label"])

# Check shapes
print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# Splitting Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=57)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)     # trained model

# Predictions
y_pred_lr = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_lr)
print("V2 Accuracy:", accuracy)

# Evaluation
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix:\n", cm_lr)

# Classification report
cr_lr = classification_report(y_test, y_pred_lr)
print("\nClassification Report:\n",cr_lr)

# Genetic ALgorithm
import pygad

# Fitness Calculation
def fitness_func(ga_instance, solution, solution_idx):
    # Select features based on chromosome
    selected_features = np.where(solution == 1)[0]

    # Avoid empty feature set
    if len(selected_features) == 0:
        return 0

    # Select only chosen columns
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Train temporary model
    temp_model = LogisticRegression(max_iter=300)
    temp_model.fit(X_train_selected, y_train)

    # Predict
    y_pred_temp = temp_model.predict(X_test_selected)

    # Return accuracy as fitness
    return accuracy_score(y_test, y_pred_temp)

num_features = X_train.shape[1]

ga_instance = pygad.GA(num_generations=10, num_parents_mating=4, fitness_func=fitness_func, sol_per_pop=8,
                        num_genes=num_features, gene_type=int, init_range_low=0, init_range_high=2,mutation_percent_genes=10)

print("Running Genetic Algorithm...")
ga_instance.run()
print("GA completed!")

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best GA Accuracy:", solution_fitness)

selected_features = np.where(solution == 1)[0]
print("Number of selected features:", len(selected_features))