import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder


def preprocess_data(data):
    # Encode non-numeric columns
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])
    return data


def evaluate_knn_classifier(X_train, X_test, y_train, y_test, k):
    # Train kNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)

    # Test the classifier
    predictions = knn_classifier.predict(X_test)

    # Evaluate performance
    confusion_mat = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)
    f1 = f1_score(y_test, predictions, average=None)

    return confusion_mat, precision, recall, f1


def hyperparameter_tuning(X_train, y_train):
    # Define the parameter grid
    param_grid = {'n_neighbors': np.arange(1, 21)}

    # Create kNN classifier
    knn_classifier = KNeighborsClassifier()

    # Perform Randomized Search CV
    random_search = RandomizedSearchCV(knn_classifier, param_distributions=param_grid, n_iter=10, cv=5)
    random_search.fit(X_train, y_train)

    # Get the best k value
    best_k = random_search.best_params_['n_neighbors']

    return best_k


def main():
    # Preprocess data
    # Load your project data (replace 'your_data.csv' with your actual file)
    data = pd.read_csv(r"C:\Users\sripriya konjarla\Downloads\Kavinya\AISP.csv")
    data = preprocess_data(data)

    # Split data into features (X) and target (y)
    X = data.drop('Label', axis=1)  # Adjust column names as per your data
    y = data['Label']  # Adjust column name as per your data

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Choose k value using hyperparameter tuning
    best_k = hyperparameter_tuning(X_train, y_train)

    # Evaluate kNN classifier with the best k
    confusion_mat, precision, recall, f1 = evaluate_knn_classifier(X_train, X_test, y_train, y_test, best_k)

    # Print and analyze results
    print("Confusion Matrix:")
    print(confusion_mat)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


if __name__ == "__main__":
    main()