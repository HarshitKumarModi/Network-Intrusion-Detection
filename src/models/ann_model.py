import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

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


def create_model(input_dim):

    model = Sequential()

    model.add(
        Dense(
            64,
            activation="relu",
            input_shape=(input_dim,)
        )
    )

    model.add(
        Dense(
            32,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(model, X_train, y_train):

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history


def predict(model, X_test):

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = (probabilities > 0.5).astype(int)

    return predictions.flatten()


def plot_training(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"], label="Training")

    plt.plot(history.history["val_accuracy"], label="Validation")

    plt.legend()

    plt.title("ANN Accuracy")

    plt.savefig("images/ann_accuracy.png")

    plt.close()

    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"], label="Training")

    plt.plot(history.history["val_loss"], label="Validation")

    plt.legend()

    plt.title("ANN Loss")

    plt.savefig("images/ann_loss.png")

    plt.close()


def save_model(model):

    model.save(
        "src/models/artifacts/ann_model.keras"
    )

    print("\nANN Model Saved Successfully!")


def main():

    print("=" * 60)

    print("Artificial Neural Network")

    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()

    y_train, y_test = prepare_target(
        y_train,
        y_test
    )

    model = create_model(
        X_train.shape[1]
    )

    model, history = train_model(
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

    plot_training(
        history
    )

    save_model(
        model
    )


if __name__ == "__main__":
    main()