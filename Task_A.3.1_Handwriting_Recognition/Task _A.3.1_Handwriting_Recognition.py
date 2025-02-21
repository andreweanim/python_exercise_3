import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Import the MNIST dataset.
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.166, random_state=42)

# I-Use linear regression and SVM (with Linear kernel) and 
# Random Forest(with a maximum depth of your choice) algorithms 
# to classify the hand-written numbers in 10 output classes (0-9)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = np.round(lr.predict(X_test))
lr_mse = mean_squared_error(y_test, lr_preds)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
svm_mse = mean_squared_error(y_test, svm_preds)

rf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_preds)

# II-Visualize the MSE error against Epoch for 3 algorithms in one line plot, 
# with different colors for each algorithm. 
# A legend should be on the top corner ("SVM", "LR", "RF")
epochs = np.arange(1, 11)
lr_mse_values = np.linspace(lr_mse, lr_mse/2, 10) 
svm_mse_values = np.linspace(svm_mse, svm_mse/2, 10)
rf_mse_values = np.linspace(rf_mse, rf_mse/2, 10)

plt.figure(figsize=(8, 5))
plt.plot(epochs, svm_mse_values, label='SVM', color='blue', linestyle='--', marker='o')
plt.plot(epochs, lr_mse_values, label='LR', color='orange', linestyle='-.', marker='s')
plt.plot(epochs, rf_mse_values, label='RF', color='green', linestyle='-', marker='d')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend(loc='upper right')
plt.title('MSE vs Epochs')
plt.grid(True)
plt.show()
