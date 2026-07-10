import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)


def evaluate_model(model, X_test, y_test, predictions):

    accuracy = accuracy_score(y_test, predictions)

    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print("\nAccuracy")
    print(f"{accuracy:.4f}")

    precision = precision_score(y_test, predictions)

    print("\nPrecision")
    print(f"{precision:.4f}")

    recall = recall_score(y_test, predictions)

    print("\nRecall")
    print(f"{recall:.4f}")

    f1 = f1_score(y_test, predictions)

    print("\nF1 Score")
    print(f"{f1:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()

    plt.savefig("images/confusion_matrix.png")

    plt.close()

    # -------- ROC CURVE -------- #

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)[:, 1]

    else:

        probabilities = model.predict(
            X_test,
            verbose=0
        ).flatten()

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities
    )

    plt.savefig("images/roc_curve.png")

    plt.close()

    roc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\nROC AUC")
    print(f"{roc:.4f}")