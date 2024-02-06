import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def knn_classifier(X_train, y_train, X_test, k):
    predictions = []

    for test_point in X_test:
        
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]

        k_neighbors_indices = np.argsort(distances)[:k]

        k_neighbors_labels = [y_train[i] for i in k_neighbors_indices]

        predicted_label = max(set(k_neighbors_labels), key=k_neighbors_labels.count)
        predictions.append(predicted_label)

    return predictions

X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1]) 

X_test = np.array([[2, 3], [4, 5]])
k_value = 2

predictions = knn_classifier(X_train, y_train, X_test, k_value)

print(f"Predictions for k={k_value}: {predictions}")
