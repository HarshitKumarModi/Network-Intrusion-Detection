import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from src.visualization.evaluate_model import evaluate_model
from src.visualization.feature_importance import plot_feature_importance


X_TRAIN = "data/processed/X_train.csv"
X_TEST = "data/processed/X_test.csv"

Y_TRAIN = "data/processed/y_train.csv"
Y_TEST = "data/processed/y_test.csv"

PARAM_GRID = {

    "criterion": [
        "gini",
        "entropy"
    ],

    "max_depth": [
        5,
        10,
        15,
        20
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ]

}



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

    model = DecisionTreeClassifier(
        random_state=42
    )

    grid_search = GridSearchCV(

        estimator=model,

        param_grid=PARAM_GRID,

        cv=5,

        scoring="accuracy",

        n_jobs=-1

    )

    return grid_search


def train_model(model, X_train, y_train):

    model.fit(
        X_train,
        y_train
    )

    return model


def predict(model, X_test):

    predictions = model.predict(
        X_test
    )

    return predictions


def save_model(model):

    joblib.dump(
        model.best_estimator_,
        "src/models/artifacts/decision_tree.pkl"
    )

    print("\nDecision Tree Saved Successfully!")


def main():

    print("=" * 60)
    print("Decision Tree")
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
        model,
        X_test,
        y_test,
        predictions
    )

    plot_feature_importance(
    model.best_estimator_,
    X_train.columns
)

    save_model(model)


if __name__ == "__main__":
    main()