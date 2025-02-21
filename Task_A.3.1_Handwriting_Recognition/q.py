import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)  # Normalize and convert target to int

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
log_reg = LogisticRegression(max_iter=100, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
svm = SVC(kernel='linear')
random_forest = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)

# Train models and record MSE
models = {'LR': log_reg, 'SVM': svm, 'RF': random_forest}
mse_values = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_values[name] = mean_squared_error(y_test, y_pred)

# Plot MSE comparison
plt.figure(figsize=(8, 5))
plt.bar(mse_values.keys(), mse_values.values(), color=['blue', 'red', 'green'])
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.title('MSE Comparison of Models on MNIST')
plt.show()
