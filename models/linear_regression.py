import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_linear_regression():

    # 1. Load dataset
    df = pd.read_csv("data/students.csv")

    # 2. Separate features and target
    X = df[["study_hours", "attendance", "previous_score"]]
    y = df["exam_score"]

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    # 4. Create model
    model = LinearRegression()

    # 5. Train model
    model.fit(X_train, y_train)

    # 6. Make predictions
    predictions = model.predict(X_test)

    # 7. Evaluate model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n===== LINEAR REGRESSION =====")
    print("Actual scores:    ", list(y_test))
    print("Predicted scores: ", predictions.round(2))

    print("Mean Absolute Error:", round(mae, 2))
    print("Mean Squared Error:", round(mse, 2))
    print("R2 Score:", round(r2, 2))

    return model


if __name__ == "__main__":
    train_linear_regression()