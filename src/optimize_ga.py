import numpy as np
import joblib
import pygad

from preprocess import load_and_preprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Load data
data = load_and_preprocess("../data/train.tsv")

# 2. TF-IDF (same as train.py)
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(data["full_text"]).toarray()

# 3. Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(data["label"])

# 4. Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


# GA Fitness Function
def fitness_func(ga_instance, solution, solution_idx):
    selected_features = np.where(solution == 1)[0]

    if len(selected_features) == 0:
        return 0

    X_train_sel = X_train[:, selected_features]
    X_test_sel = X_test[:, selected_features]

    model = LogisticRegression(max_iter=300,class_weight='balanced')
    model.fit(X_train_sel, y_train)

    y_pred = model.predict(X_test_sel)

    return accuracy_score(y_test, y_pred)


# GA setup
num_features = X_train.shape[1]

ga_instance = pygad.GA(
    num_generations=10,
    num_parents_mating=4,
    sol_per_pop=8,
    num_genes=num_features,
    gene_type=int,
    init_range_low=0,
    init_range_high=2,
    mutation_percent_genes=10,
    fitness_func=fitness_func
)

print("Running GA...")
ga_instance.run()
print("GA completed!")

# Best solution
solution, solution_fitness, _ = ga_instance.best_solution()

print("\nBest GA Accuracy:", solution_fitness)

selected_features = np.where(solution == 1)[0]
print("Selected features:", len(selected_features))

# Save selected features
np.save("../artifacts/selected_features.npy", selected_features)

print("Saved selected features!")