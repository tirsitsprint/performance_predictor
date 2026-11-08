import pandas as pd
from models.nearest_neighbor import train_knn_model
from models.linear_regression import train_linear_regression
from models.logistic_regression import train_logistic_regression


def main():

    print("=" * 50)
    print("     STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("=" * 50)

    # Train all three models
    knn_model = train_knn_model()

    linear_model = train_linear_regression()

    logistic_model = train_logistic_regression()

    print("\n===== PREDICT A NEW STUDENT =====")

    # Collect input from the user
    study_hours = float(input("Enter study hours: "))
    attendance = float(input("Enter attendance percentage: "))
    previous_score = float(input("Enter previous score: "))

    # Prepare the new student's information
  

    new_student = pd.DataFrame(
       [[study_hours, attendance, previous_score]],
       columns=["study_hours", "attendance", "previous_score"]
        )

    # Make predictions
    knn_prediction = knn_model.predict(new_student)[0]

    score_prediction = linear_model.predict(new_student)[0]

    logistic_prediction = logistic_model.predict(new_student)[0]

    # Display predictions
    print("\n===== PREDICTION RESULTS =====")

    print("KNN predicted result:", knn_prediction)

    print(
        "Linear Regression predicted exam score:",
        round(score_prediction, 2)
    )

    print("Logistic Regression predicted result:", logistic_prediction)

    print("\nNote: These are educational model predictions,")
    print("not reliable judgments of a real student's ability.")


if __name__ == "__main__":
    main()