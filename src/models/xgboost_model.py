import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.visualization.evaluate_model import evaluate_model
from src.visualization.feature_importance import plot_feature_importance


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

    model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
    )

    return model


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

        model,

        "src/models/artifacts/xgboost.pkl"

    )

    print("\nXGBoost Model Saved Successfully!")

def tune_model(model, X_train, y_train):

    grid = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid


PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
    "learning_rate": [0.1, 0.3]
}

def main():

    print("=" * 60)
    print("XGBoost")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()

    y_train, y_test = prepare_target(
        y_train,
        y_test
    )

    model = create_model()

    model = tune_model(
    model,
    X_train,
    y_train
    )
    
    print("\nBest Parameters")
    print(model.best_params_)

    print("\nBest Cross Validation Score")
    print(model.best_score_)

    

    predictions = model.predict(X_test)

    evaluate_model(
    model.best_estimator_,
    X_test,
    y_test,
    predictions
    )

    plot_feature_importance(
    model.best_estimator_,
    X_train.columns
    )

    save_model(
    model.best_estimator_
)


if __name__ == "__main__":
    main()