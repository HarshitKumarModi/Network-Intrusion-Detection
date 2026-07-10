from src.data.load_data import load_training_data


OUTPUT_PATH = "data/processed/train_binary.csv"





def encode_attack_column(df):

    df["attack"] = df["attack"].apply(
        lambda x: 0 if x == "normal" else 1
    )

    return df

def save_dataset(df):

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nProcessed dataset saved successfully!")

def main():

    print("=" * 60)
    print("Binary Label Encoding")
    print("=" * 60)

    df = load_training_data()

    print("\nOriginal Labels\n")
    print(df["attack"].value_counts())

    df = encode_attack_column(df)

    print("\nBinary Labels\n")
    print(df["attack"].value_counts())

    save_dataset(df)


if __name__ == "__main__":
    main()