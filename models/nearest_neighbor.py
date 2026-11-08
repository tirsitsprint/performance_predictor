import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def train_knn_model():

    # 1. Load the dataset
    df = pd.read_csv("data/students.csv")

    # 2. Separate features and target
    X = df[["study_hours", "attendance", "previous_score"]]
    y = df["result"]

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # 4. Create the KNN model
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=3)
    )

    # 5. Train the model
    model.fit(X_train, y_train)

    # 6. Make predictions
    predictions = model.predict(X_test)

    # 7. Evaluate the model
    accuracy = accuracy_score(y_test, predictions)

    print("\n===== KNN CLASSIFICATION =====")
    print("Actual results:    ", list(y_test))
    print("Predicted results: ", list(predictions))
    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return model


if __name__ == "__main__":
    train_knn_model()