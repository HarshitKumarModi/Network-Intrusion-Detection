import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, feature_names):

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df = importance_df.head(20)

    plt.figure(figsize=(12,8))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Top 20 Feature Importances")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig("images/feature_importance.png")

    plt.close()