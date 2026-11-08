# Student Performance Prediction System

## About the Project

The Student Performance Prediction System is a beginner-friendly machine learning project that demonstrates how different machine learning algorithms can be used to analyze fictional student data and make predictions.

The project uses three machine learning algorithms:

1. **K-Nearest Neighbors (KNN):** Predicts whether a student will Pass or Fail by comparing the student with similar examples.
2. **Linear Regression:** Predicts a student's numerical exam score.
3. **Logistic Regression:** Predicts whether a student will Pass or Fail using classification.

This project was developed for learning and demonstrating fundamental machine learning concepts.

## Project Objectives

- Understand how to prepare a dataset for machine learning.
- Learn the difference between features and target variables.
- Practice training and testing machine learning models.
- Understand classification and regression.
- Compare the predictions and performance of different algorithms.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Dataset

The project uses a small, fictional dataset containing 24 student records.

| Feature | Description |
|---|---|
| study_hours | Number of hours spent studying |
| attendance | Student attendance percentage |
| previous_score | Previous examination score |
| exam_score | Current examination score |
| result | Pass or Fail |

The models use the following inputs:

- Study hours
- Attendance
- Previous score

The prediction targets are:

- **Exam score** for Linear Regression.
- **Pass/Fail result** for KNN and Logistic Regression.

The dataset contains fictional examples and should not be used to make real educational decisions.

## Machine Learning Algorithms

### 1. K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm used for classification.

It finds the nearest examples in the training dataset and predicts a class based on their labels.

In this project, KNN uses three nearest neighbors to predict Pass or Fail.

### 2. Linear Regression

Linear Regression is a supervised machine learning algorithm used for predicting numerical values.

It learns relationships between input features and a numerical target.

In this project, it predicts examination scores.

### 3. Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for classification.

It estimates the probability of belonging to a particular class and predicts Pass or Fail.

## Project Structure

```text
performance/
│
├── data/
│   └── students.csv
│
├── models/
│   ├── nearest_neighbor.py
│   ├── linear_regression.py
│   └── logistic_regression.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Navigate into the project directory

```bash
cd performance
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install the required libraries

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Run the main program from the project's root directory:

```bash
python main.py
```

The program will:

1. Load the student dataset.
2. Train the three machine learning models.
3. Evaluate their performance.
4. Ask for new student information.
5. Display the predictions from all three models.

### Example Input

```text
Enter study hours: 7
Enter attendance percentage: 85
Enter previous score: 75
```

The program displays:

- KNN predicted result.
- Linear Regression predicted exam score.
- Logistic Regression predicted result.

## Model Evaluation

The project uses different evaluation metrics depending on the type of machine learning task.

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score

### Regression Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

The models are evaluated using a train-test split, with 75% of the data used for training and 25% for testing.

## Important Limitations

- The dataset is small and contains only 24 fictional records.
- Model performance on this dataset does not guarantee performance on real-world student data.
- Predictions are educational demonstrations, not reliable judgments of student ability.
- More representative data and additional testing would be needed for any real-world application.

## Learning Outcomes

Through this project, I practiced:

- Working with CSV datasets using Pandas.
- Preparing features and target variables.
- Splitting data into training and testing sets.
- Training supervised machine learning models.
- Applying classification and regression algorithms.
- Evaluating machine learning performance.
- Organizing a Python machine learning project.

## Author

**Tirsit**

Beginner Machine Learning Project
