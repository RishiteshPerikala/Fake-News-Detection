import pandas as pd

# Column names for LIAR dataset
columns = ["id", "label", "statement", "subject", "speaker","speaker_job", "state", "party","barely_true", "false", "half_true", "mostly_true", "pants_on_fire", "context"]

# Load dataset
data = pd.read_csv("data/train.tsv", sep="\t", header=None, names=columns)

# Show first rows
print(data.head())
print("\nColumns:\n", data.columns)

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