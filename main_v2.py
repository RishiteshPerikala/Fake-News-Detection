import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Train model

# -------- Original Model --------
model_original = LogisticRegression(max_iter=500, solver='lbfgs')

model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)

print("\n=== ORIGINAL MODEL ===")
print("Accuracy:", accuracy_score(y_test, y_pred_original))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_original))
print("Classification Report:\n", classification_report(y_test, y_pred_original))

# -------- Balanced Model --------
model = LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced')

model.fit(X_train, y_train)

# Predictions
y_pred_lr = model.predict(X_test)

print("\n=== BALANCED MODEL ===")
print("Accuracy:",accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Classification report
cr_lr = classification_report(y_test, y_pred_lr)

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

ga_instance = pygad.GA(
    num_generations=10, num_parents_mating=4, fitness_func=fitness_func, sol_per_pop=8, num_genes=num_features, 
    gene_type=int,  init_range_low=0, init_range_high=2, mutation_percent_genes=10)

#print("Running Genetic Algorithm...")
#ga_instance.run()
#print("GA completed!")

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best GA Accuracy:", solution_fitness)

selected_features = np.where(solution == 1)[0]
print("Number of selected features:", len(selected_features))

np.save("selected_features.npy", selected_features) # save GA

selected_features = np.load("selected_features.npy")

# Select optimized features
X_train_ga = X_train[:, selected_features]
X_test_ga = X_test[:, selected_features]

# Implementing ANN
from sklearn.neural_network import MLPClassifier

# Create ANN model
model_ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True, random_state=42)

# Train Model
print("\nTraining ANN model...")
model_ann.fit(X_train_ga, y_train)
print("ANN training completed!")

# Accuracy
y_pred_ann = model_ann.predict(X_test_ga)
accuracy_ann = accuracy_score(y_test, y_pred_ann)
print("ANN Accuracy:", accuracy_ann)

# Metrics
print("\nANN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ann))
print("\nANN Classification Report:\n", classification_report(y_test, y_pred_ann))

# Comparing LR and ANN using Plot
#1
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_ann = accuracy_score(y_test, y_pred_ann)

models = ['Balanced LR', 'ANN']
accuracies = [acc_lr, acc_ann]

plt.figure()
plt.bar(models, accuracies)
plt.title("Accuracy Comparison: LR vs ANN")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

#2
from sklearn.metrics import precision_score, recall_score, f1_score

# LR scores
prec_lr = precision_score(y_test, y_pred_lr, average='macro')
rec_lr = recall_score(y_test, y_pred_lr, average='macro')
f1_lr = f1_score(y_test, y_pred_lr, average='macro')

# ANN scores
prec_ann = precision_score(y_test, y_pred_ann, average='macro')
rec_ann = recall_score(y_test, y_pred_ann, average='macro')
f1_ann = f1_score(y_test, y_pred_ann, average='macro')

labels = ['Precision', 'Recall', 'F1-score']

lr_scores = [prec_lr, rec_lr, f1_lr]
ann_scores = [prec_ann, rec_ann, f1_ann]

x = range(len(labels))

plt.figure()
plt.plot(x, lr_scores, marker='o', label='Balanced LR')
plt.plot(x, ann_scores, marker='o', label='ANN')

plt.xticks(x, labels)
plt.title("Macro Metrics Comparison: LR vs ANN")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.legend()
plt.show()

#3

def plot_cm(cm, title):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    
    ticks = np.arange(len(cm))
    plt.xticks(ticks)
    plt.yticks(ticks)
    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# LR
plot_cm(confusion_matrix(y_test, y_pred_lr), "Balanced LR Confusion Matrix")

# ANN
plot_cm(confusion_matrix(y_test, y_pred_ann), "ANN Confusion Matrix")

# LR is better than ANN

# Applying Fuzzy Logic

# define fuzzy function
def fuzzy_op(p):
    if p < 0.25:
        return "Highly Fake"
    elif p < 0.45:
        return "Mostly Fake"
    elif p < 0.55:
        return "Uncertain"
    elif p < 0.75:
        return "Mostly Real"
    else:
        return "Highly Real"
    
# calculating probability
y_prob_lr = model.predict_proba(X_test)    #uses sigmoid to calculate probability

# get first 10 fuzzy outputs 
#for i in range(10):
#    prob_lr = np.max(y_prob_lr[i])
#    print("Probability: ",prob_lr)             # prints probabilities
#    print("Fuzzy o/p: ",fuzzy_op(prob_lr))     # prints fuzzy outputs
#   print()
#
#print("\n----- Test Your Own Input -----")

#user_input = input("Enter news: ")

# Convert using same vectorizer
#input_vector = vectorizer.transform([user_input]).toarray()

# Predict probability
#prob = np.max(model.predict_proba(input_vector)[0])

#print("Probability:", prob)
#print("Fuzzy Output:", fuzzy_op(prob))