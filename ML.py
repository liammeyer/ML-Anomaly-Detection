import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load Data
attack = pd.read_csv('Attack.csv', low_memory=False)
patient = pd.read_csv('patientMonitoring.csv', low_memory=False)
environment = pd.read_csv('environmentMonitoring.csv')

# Preprocess and handle missing data if needed
# attack.replace([np.inf, -np.inf], np.nan, inplace=True)
# patient.replace([np.inf, -np.inf], np.nan, inplace=True)
# environment.replace([np.inf, -np.inf], np.nan, inplace=True)
# attack.dropna(inplace=True)
# patient.dropna(inplace=True)
# environment.dropna(inplace=True)

# Combine patient and attack data
combinedPatientAttack = pd.concat([patient, attack], axis=0)

# Prepare Data
X = combinedPatientAttack.drop('label', axis=1)  # Drop the target column to create a feature set
X = X.select_dtypes(exclude='object')  # This excludes all columns of type 'object', typically strings

y = combinedPatientAttack['label']  # Keep only the target column

# Data feature scaling, using sklearn
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize MLPClassifier

'''
#2 layers - 2 neurons in each - 51% accuracy
mlp = MLPClassifier(hidden_layer_sizes=(2, 2), activation='relu', solver='adam', random_state=1, verbose=True, early_stopping=True, max_iter=300)
#3 layers - 50 in first, 20 in second, and 5 in third - 99.98% accuracy
mlp = MLPClassifier(hidden_layer_sizes=(50, 20, 5), activation='relu', solver='adam', random_state=1, verbose=True, early_stopping=True, max_iter=300)
#3 layers - 2 neurons in each - 69% accuracy
mlp = MLPClassifier(hidden_layer_sizes=(2, 2, 2), activation='relu', solver='adam', random_state=1, verbose=True, early_stopping=True, max_iter=300)
#3 layers - 2 neurons in each - 92% accuracy
mlp = MLPClassifier(hidden_layer_sizes=(5, 3, 2), activation='relu', solver='adam', random_state=1, verbose=True, early_stopping=True, max_iter=300)
#3 layers - 5,3,2 neurons in each - tanh activation function - 94% accuracy
mlp = MLPClassifier(hidden_layer_sizes=(5, 3, 2), activation='tanh', solver='adam', random_state=1, verbose=True, early_stopping=True, max_iter=300)
#3 layers - 8,4,2 neurons in each - 95% accuracy
mlp = MLPClassifier(hidden_layer_sizes=(8, 4, 2), activation='tanh', solver='adam', random_state=1, verbose=True, early_stopping=True, max_iter=300)
#3 layers - 8,4,2 neurons in each - logistic activation function, performs similar to tanh - 95% accuracy
mlp = MLPClassifier(hidden_layer_sizes=(8, 4, 2), activation='logistic', solver='adam', random_state=1, verbose=True, early_stopping=True, max_iter=300)
'''

# Use a tanh activation function because its useful when there is data centered around zero which is true in our case
#3 layers - 8,4,2 neurons in each - 95.2% accuracy
mlp = MLPClassifier(hidden_layer_sizes=(16, 8, 2), activation='tanh', solver='adam', random_state=1, verbose=True, early_stopping=True, max_iter=300)

# Re-train using the training data
mlp.fit(X_train, y_train)  # Use X_train and y_train here

# Predict the labels for the test set
predictions = mlp.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)
