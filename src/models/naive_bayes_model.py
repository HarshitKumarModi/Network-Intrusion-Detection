import pandas as pd
import joblib

from sklearn.naive_bayes import GaussianNB

from src.visualization.evaluate_model import evaluate_model

X_TRAIN = "data/processed/X_train.csv"
X_TEST = "data/processed/X_test.csv"

Y_TRAIN = "data/processed/y_train.csv"
Y_TEST = "data/processed/y_test.csv"

def load_data():

    X_train = pd.read_csv(X_TRAIN)

    X_test = pd.read_csv(X_TEST)

    y_train = pd.read_csv(Y_TRAIN)

    y_test = pd.read_csv(Y_TEST)

    return X_train, X_test, y_train, y_test

def prepare_target(y_train, y_test):

    y_train = y_train.squeeze()

    y_test = y_test.squeeze()

    return y_train, y_test

def create_model():

    model = GaussianNB()

    return model

def train_model(model, X_train, y_train):

    model.fit(
        X_train,
        y_train
    )

    return model

def predict(model, X_test):

    predictions = model.predict(X_test)

    return predictions

def save_model(model):

    joblib.dump(
        model,
        "src/models/artifacts/naive_bayes.pkl"
    )

    print("\nNaive Bayes Model Saved Successfully!")

def main():

    print("=" * 60)
    print("Naive Bayes")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()

    y_train, y_test = prepare_target(
        y_train,
        y_test
    )

    model = create_model()

    model = train_model(
        model,
        X_train,
        y_train
    )

    predictions = predict(
        model,
        X_test
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        predictions
    )

    save_model(model)

if __name__ == "__main__":
    main()