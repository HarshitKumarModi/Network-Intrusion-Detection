import pandas as pd

from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/train_final.csv"

def load_dataset():

    df = pd.read_csv(DATA_PATH)

    return df

def split_features_target(df):

    X = df.drop(columns=["attack"])

    y = df["attack"]

    return X, y

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )

    return X_train, X_test, y_train, y_test

def save_dataset(df, path):

    df.to_csv(

        path,

        index=False

    )

def main():

    print("=" * 60)

    print("Train Test Split")

    print("=" * 60)

    df = load_dataset()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_dataset(

        X,

        y

    )

    print("\nTraining Samples")

    print(X_train.shape)

    print("\nTesting Samples")

    print(X_test.shape)

    save_dataset(

        X_train,

        "data/processed/X_train.csv"

    )

    save_dataset(

        X_test,

        "data/processed/X_test.csv"

    )

    save_dataset(

        y_train.to_frame(),

        "data/processed/y_train.csv"

    )

    save_dataset(

        y_test.to_frame(),

        "data/processed/y_test.csv"

    )

    print("\nDatasets Saved Successfully!")

if __name__ == "__main__":
    main()