import pandas as pd
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

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

    model = KNeighborsClassifier()

    return model

PARAM_GRID = {

    "n_neighbors": [3, 5],

    "weights": ["uniform", "distance"],

    "metric": ["euclidean"]

}

def train_model(model, X_train, y_train):

    grid = GridSearchCV(

        estimator=model,

        param_grid=PARAM_GRID,

        cv=3,

        scoring="accuracy",

        n_jobs=-1

    )

    grid.fit(

        X_train,

        y_train

    )

    return grid
def predict(model, X_test):

    predictions = model.predict(X_test)

    return predictions

def save_model(model):

    joblib.dump(

        model,

        "src/models/artifacts/knn.pkl"

    )

    print("\nKNN Model Saved Successfully!")

def main():

    print("=" * 60)
    print("K-Nearest Neighbors")
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

    print("\nBest Parameters")
    print(model.best_params_)

    print("\nBest Cross Validation Score")
    print(model.best_score_)

    predictions = predict(
        model,
        X_test
    )

    evaluate_model(
        model.best_estimator_,
        X_test,
        y_test,
        predictions
    )

    save_model(
        model.best_estimator_
    )

if __name__ == "__main__":
    main()