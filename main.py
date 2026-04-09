import pandas as pd
import nltk
import string
from nltk.corpus import stopwords

#download stopwords
nltk.download('stopwords')

# Define column names (IMPORTANT)
columns = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state", "party",
    "barely_true", "false", "half_true",
    "mostly_true", "pants_on_fire", "context"
]

# Load dataset with column names
data = pd.read_csv("data/train.tsv", sep="\t", names=columns)

# Now select only needed columns
data = data[["label", "statement"]]

#convert labels to binary
def convert_label(x):
    if x in ["true","mostly-true"]:
        return 1
    else:
        return 0
    
#convert entire column
data["label"] = data["label"].apply(convert_label)

#load stopwords
stop_words = set(stopwords.words('english'))

#Preprocessing function
def preprocess(text):
    text = text.lower()  # lowercase text
    text = text.translate(str.maketrans('','',string.punctuation))  #remove punctuations
    words = text.split()    #split into words(tokenization)
    words = [word for word in words if word not in stop_words]  # remove stopwords

    return " ".join(words)

#apply preprocess
data["statement"] = data["statement"].apply(preprocess)

# Display
print(data.head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["statement"]).toarray()   #convert text to numbers

y = data["label"]   #assigning y as labels i.e 0 or 1

#print shape
print("Feature shape:",X.shape)
print("Labels shape:",y.shape)

#Training & Testing Model
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=57)

# Print shapes
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Train model
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(class_weight='balanced')    # updated model (treats fake and real equally)
model_lr.fit(X_train,y_train)  # trained model

print("Logistic Regression Model is trained Successfully!")

# Predict the data
y_pred_lr = model_lr.predict(X_test)  # predicts values as 0's and 1's

print("Predictions: ",y_pred_lr[:10])      # shows first 10 prediction values
print("Actual: ",y_test[:10].values)   # shows first 10 actual values

# calculate Accuracy and add Confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

acc_lr = accuracy_score(y_test,y_pred_lr)    # comparing both
print("Accuracy: ",acc_lr)

cm_lr = confusion_matrix(y_test,y_pred_lr)    #confusion matrix
print("Confusion Matrix:\n",cm_lr)

# Applying Fuzzy Logic

# define fuzzy function
def fuzzy_op(p):
    if p < 0.2:
        return "Highly Fake"
    elif p < 0.4:
        return "Mostly Fake"
    elif p < 0.6:
        return "Uncertain"
    elif p < 0.8:
        return "Mostly Real"
    else:
        return "Highly Real"
    
# calculating probability
y_prob_lr = model_lr.predict_proba(X_test)    #uses sigmoid to calculate probability

real_prob_lr = y_prob_lr[:,1] # get real news's probability

# get first 10 fuzzy outputs 
for i in range(10):
    print("Probability: ",real_prob_lr[i])             # prints probabilities
    print("Fuzzy o/p: ",fuzzy_op(real_prob_lr[i]))     # prints fuzzy outputs
    print()

# Testing model by entering user input
# inp = input("Enter news: ")

# converting text into scores
#inp_vector = vectorizer.transform([inp]).toarray()

#predicting probability for user input text
#prob_lr= model_lr.predict_proba(inp_vector)[0][1]

# print probability and fuzzy o/p
#print("Probability: ",prob_lr)
#print("Fuzzy Output: ",fuzzy_op(prob_lr))

#ANN
from sklearn.neural_network import MLPClassifier

model_ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)

model_ann.fit(X_train, y_train)

print("ANN model trained!")

y_pred_ann = model_ann.predict(X_test)  # predicting values as 0's & 1's 

acc_ann = accuracy_score(y_test, y_pred_ann)    # comparing both 
print("ANN Accuracy:", acc_ann)

cm_ann = confusion_matrix(y_test,y_pred_ann)    #confusion matrix
print("Confusion Matrix:\n",cm_ann)

# calculating Probability
y_prob_ann = model_ann.predict_proba(X_test)    #uses sigmoid to calculate probability

real_prob_ann = y_prob_ann[:, 1]    # get real news's probability

# get first 10 fuzzy outputs
for i in range(10):
    print("ANN Probability:", real_prob_ann[i])
    print("Fuzzy Output:", fuzzy_op(real_prob_ann[i]))
    print()
    
# Testing model by entering user input
inp = input("Enter news: ")

# converting text into scores
inp_vector = vectorizer.transform([inp]).toarray()

#predicting probability for user input text
prob_ann= model_ann.predict_proba(inp_vector)[0][1]

# print probability and fuzzy o/p
print("Probability: ",prob_ann)
print("Fuzzy Output: ",fuzzy_op(prob_ann))