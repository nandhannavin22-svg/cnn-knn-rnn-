# Import libraries
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------------
# STEP 1: Load MNIST Dataset
# -----------------------------------

mnist = fetch_openml('mnist_784', version=1)

X = mnist.data
y = mnist.target

# -----------------------------------
# STEP 2: Normalize Data
# -----------------------------------

X = X / 255.0

# -----------------------------------
# STEP 3: Split Data
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# STEP 4: Create KNN Model
# -----------------------------------

knn = KNeighborsClassifier(n_neighbors=3)

# -----------------------------------
# STEP 5: Train Model
# -----------------------------------

knn.fit(X_train, y_train)

# -----------------------------------
# STEP 6: Predict
# -----------------------------------

y_pred = knn.predict(X_test)

# -----------------------------------
# STEP 7: Evaluate
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nKNN Accuracy:", accuracy)

# -----------------------------------
# STEP 8: Save Model
# -----------------------------------

joblib.dump(knn, "knn_model.joblib")

print("\nModel saved as knn_model.joblib")