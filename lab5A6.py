import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\sripriya konjarla\Downloads\Kavinya\AISP.csv")  # Replace with the actual path

# Step 2: Data preprocessing
X = df[['Question', 'Answers']]  # Features
y = df['Label']  # Target variable

# Map string labels to numerical values
label_map = {"Wrong": 0, "Correct": 1, "Partially Correct": 2}
y = y.map(label_map)

# Step 3: Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Training data visualization (Exercise A3)
plt.figure(figsize=(10, 6))
plt.scatter(X_train['Question'], X_train['Answers'], c=y_train, cmap=ListedColormap(['blue', 'red', 'green']))
plt.title('Training Data')
plt.xlabel('Question')
plt.ylabel('Answers')
plt.show()

# Step 5: Generate test set (Exercise A4)
# Since we're not generating new test data, skip this step

# Step 6: kNN classification (Exercise A4)
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Step 7: Observe class boundary lines (Exercise A5)
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Question'], X_test['Answers'], c=y_pred, cmap=ListedColormap(['blue', 'red', 'green']))
plt.title(f'Test Data Output (k={k})')
plt.xlabel('Question')
plt.ylabel('Answers')
plt.show()