import joblib
from tensorflow.keras.models import load_model


MODEL_PATHS = {

    "Logistic Regression":
    "src/models/artifacts/logistic_regression.pkl",

    "Decision Tree":
    "src/models/artifacts/decision_tree.pkl",

    "Random Forest":
    "src/models/artifacts/random_forest.pkl",

    "XGBoost":
    "src/models/artifacts/xgboost.pkl",

    "Support Vector Machine":
    "src/models/artifacts/svm.pkl",

    "KNN":
    "src/models/artifacts/knn.pkl",

    "Naive Bayes":
    "src/models/artifacts/naive_bayes.pkl",

    "Artificial Neural Network":
    "src/models/artifacts/ann_model.keras"

}

def load_selected_model(model_name):

    path = MODEL_PATHS[model_name]

    if path.endswith(".keras"):

        model = load_model(path)

    else:

        model = joblib.load(path)

    return model
