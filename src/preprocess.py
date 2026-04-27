import pandas as pd

def map_labels(label):

    """
    Convert 6 LIAR labels into 3 classes:
    Fake, Uncertain, Real
    """
     
    if label in ["pants-fire", "false"]:
        return "Fake"
    elif label in ["barely-true", "half-true"]:
        return "Uncertain"
    else:
        return "Real"

def load_and_preprocess(file_path):
    """
    Loads LIAR dataset and performs preprocessing.
    Returns cleaned dataframe with 'label' and 'full_text'.
    """
    # Column names for LIAR dataset
    columns = [
        "id", "label", "statement", "subject", "speaker",
        "speaker_job", "state", "party",
        "barely_true", "false", "half_true",
        "mostly_true", "pants_on_fire", "context"
    ]

    # Load dataset
    data = pd.read_csv(file_path, sep="\t", header=None, names=columns)

    # Fill missing values
    text_columns = [
        "statement", "subject", "speaker",
        "speaker_job", "state", "party", "context"
    ]

    for col in text_columns:
        data[col] = data[col].fillna("")

    # Combine text columns
    data["full_text"] = (
        data["statement"] + " " +
        data["subject"] + " " +
        data["speaker"] + " " +
        data["speaker_job"] + " " +
        data["state"] + " " +
        data["party"] + " " +
        data["context"]
    )

    data["label"] = data["label"].apply(map_labels)

    # Keep only required columns
    data = data[["label", "full_text"]]

    return data

#test individually
if __name__ == "__main__":
    df = load_and_preprocess("../data/train.tsv")
    print(df.head())

    print("\nClass Distribution:")
    print(df["label"].value_counts())

    print("\nData shape:", df.shape)