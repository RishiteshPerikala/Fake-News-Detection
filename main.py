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
print("Labels.shape:",y.shape)