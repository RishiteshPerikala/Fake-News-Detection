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

# Display
print(data.head())