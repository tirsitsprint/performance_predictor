import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def train_logistic_regression():

    # 1. Load dataset
    df = pd.read_csv("data/students.csv")

    # 2. Separate features and target
    X = df[["study_hours", "attendance", "previous_score"]]
    y = df["result"]

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # 4. Create model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )

    # 5. Train model
    model.fit(X_train, y_train)

    # 6. Make predictions
    predictions = model.predict(X_test)

    # 7. Evaluate model
    accuracy = accuracy_score(y_test, predictions)

    print("\n===== LOGISTIC REGRESSION =====")
    print("Actual results:    ", list(y_test))
    print("Predicted results: ", list(predictions))
    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return model


if __name__ == "__main__":
    train_logistic_regression()