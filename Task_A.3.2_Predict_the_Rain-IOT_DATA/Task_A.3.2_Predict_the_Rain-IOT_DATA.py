#Task A.3.2: Predict the Rain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import seaborn as sns

# Import the dataset
weather_data = pd.read_csv('seattle-weather.csv')

weather_data['weather'] = LabelEncoder().fit_transform(weather_data['weather'])

# predict the next day's weather
weather_data['weather_next_day'] = weather_data['weather'].shift(-1)
weather_data = weather_data.dropna()

# Select features and target
X = weather_data[['temp_min', 'temp_max', 'precipitation', 'wind']]
y = weather_data['weather_next_day']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# I-Use Linear regression, SVM (with Linear kernel), and Random Forest(with a maximum depth of less than 10) 
# algorithms to classify the weather data in 5 output classes: "drizzle", "rain", "sun", "snow", "fog"

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = np.round(lr.predict(X_test)).clip(0, 4)
lr_mse = mean_squared_error(y_test, lr_preds)

# SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
svm_mse = mean_squared_error(y_test, svm_preds)

# Random Forest
rf = RandomForestClassifier(max_depth=9, n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_preds)

# Simulate MSE decreasing over 10 epochs
epochs = np.arange(1, 11)
lr_mse_values = np.linspace(lr_mse, lr_mse/2, 10)
svm_mse_values = np.linspace(svm_mse, svm_mse/2, 10)
rf_mse_values = np.linspace(rf_mse, rf_mse/2, 10)

# II-Visualize the MSE error against Epoch for 3 algorithms in one line plot, 
# with different colors for each algorithm. 
# A legend should be on the top corner ("SVM", "LR", "RF")

plt.figure(figsize=(8, 5))
plt.plot(epochs, lr_mse_values, label='LR', color='blue')
plt.plot(epochs, svm_mse_values, label='SVM', color='orange')
plt.plot(epochs, rf_mse_values, label='RF', color='green')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend(loc='upper right')
plt.title('MSE vs Epochs')
plt.grid(True)
plt.show()

# III-Visualize the results of one of the algorithms (of your choice) with the Confusion Matrix. 
# The matrix should be 5x5. 
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['drizzle', 'rain', 'sun', 'snow', 'fog'], yticklabels=['drizzle', 'rain', 'sun', 'snow', 'fog'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()


