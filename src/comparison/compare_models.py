import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras.models import load_model


def compare_models(X_test, y_test):

    models = {
        "Logistic Regression": "src/models/artifacts/logistic_regression.pkl",
        "Decision Tree": "src/models/artifacts/decision_tree.pkl",
        "Random Forest": "src/models/artifacts/random_forest.pkl",
        "Support Vector Machine": "src/models/artifacts/svm.pkl",
        "XGBoost": "src/models/artifacts/xgboost.pkl"
    }

    results = []

    for model_name, path in models.items():

        if not os.path.exists(path):
            continue

        model = joblib.load(path)

        predictions = model.predict(X_test)

        try:
            probabilities = model.predict_proba(X_test)[:, 1]
        except:
            probabilities = predictions

        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1 Score": f1_score(y_test, predictions),
            "ROC AUC": roc_auc_score(y_test, probabilities)
        })

    # ANN

    ann_path = "src/models/artifacts/ann_model.keras"

    if os.path.exists(ann_path):

        ann = load_model(ann_path)

        pred_prob = ann.predict(X_test).flatten()

        pred = (pred_prob > 0.5).astype(int)

        results.append({
            "Model": "Artificial Neural Network",
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
            "ROC AUC": roc_auc_score(y_test, pred_prob)
        })

        

    return pd.DataFrame(results)