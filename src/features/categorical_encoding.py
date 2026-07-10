import pandas as pd

INPUT_PATH = "data/processed/train_binary.csv"

OUTPUT_PATH = "data/processed/train_encoded.csv"

def load_dataset():

    df = pd.read_csv(INPUT_PATH)

    return df

def encode_categorical_columns(df):

    categorical_columns = [
        "protocol_type",
        "service",
        "flag"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns
    )

    return df

def save_dataset(df):

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nEncoded dataset saved successfully!")


def main():

    print("=" * 60)
    print("Categorical Encoding")
    print("=" * 60)

    df = load_dataset()

    print("\nBefore Encoding")

    print(df.shape)

    df = encode_categorical_columns(df)

    print("\nAfter Encoding")

    print(df.shape)

    save_dataset(df)

if __name__ == "__main__":
    main()