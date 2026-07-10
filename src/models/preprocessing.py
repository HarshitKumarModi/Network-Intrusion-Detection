import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

DATA_PATH = "data/processed/train_binary.csv"

def load_dataset():

    df = pd.read_csv(DATA_PATH)

    return df

def split_features_target(df):

    X = df.drop(columns=["attack"])

    y = df["attack"]


    return X, y

def get_categorical_columns():

    return [
        "protocol_type",
        "service",
        "flag"
    ]

def get_numerical_columns(X):

    categorical = get_categorical_columns()

    numerical = [
        col for col in X.columns
        if col not in categorical
    ]

    return numerical

def build_preprocessor(categorical, numerical):

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(handle_unknown="ignore"),

                categorical
            ),

            (
                "numerical",

                StandardScaler(),

                numerical
            )

        ]

    )

    return preprocessor

def apply_preprocessing(preprocessor, X):

    X_processed = preprocessor.fit_transform(X)

    return X_processed

def get_feature_names(preprocessor):

    return preprocessor.get_feature_names_out()

def convert_to_dataframe(X_processed, feature_names):

    df = pd.DataFrame(
        X_processed,
        columns=feature_names
    )

    return df

def save_processed_dataset(df):

    output_path = "data/processed/train_final.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print("\nProcessed dataset saved!")

def save_preprocessor(preprocessor):

    path = "src/models/artifacts/preprocessor.pkl"

    joblib.dump(preprocessor, path)

    print("\nPreprocessor saved successfully!")


def main():

    print("=" * 60)
    print("Professional Preprocessing")
    print("=" * 60)

    df = load_dataset()

    X, y = split_features_target(df)

    categorical = get_categorical_columns()

    numerical = get_numerical_columns(X)

    preprocessor = build_preprocessor(
        categorical,
        numerical
    )

    X_processed = apply_preprocessing(
        preprocessor,
        X
    )

    feature_names = get_feature_names(
        preprocessor
    )

    processed_df = convert_to_dataframe(
        X_processed,
        feature_names
    )

    processed_df["attack"] = y.values

    save_processed_dataset(
        processed_df
    )

    save_preprocessor(
        preprocessor
    )

    print("\nFinal Dataset Shape")

    print(processed_df.shape)

    print("\nPipeline Completed Successfully!")

if __name__ == "__main__":
    main()