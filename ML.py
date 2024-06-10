import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load Data
# attack = pd.read_csv('Attack.csv', low_memory=False)
# patient = pd.read_csv('patientMonitoring.csv', low_memory=False)
# environment = pd.read_csv('environmentMonitoring.csv')
temps = pd.read_csv('MLTempDataset.csv', low_memory=False)


# Preprocess and handle missing data if needed
temps.replace([np.inf, -np.inf], np.nan, inplace=True)
# patient.replace([np.inf, -np.inf], np.nan, inplace=True)
# environment.replace([np.inf, -np.inf], np.nan, inplace=True)
temps.dropna(inplace=True)
# patient.dropna(inplace=True)
# environment.dropna(inplace=True)


# Combine patient and attack data
# combinedPatientAttack = pd.concat([patient, attack], axis=0)

# Prepare Data
# X = combinedPatientAttack.drop('label', axis=1)  # Drop the target column to create a feature set


X = temps.drop('index', axis=1)
X = temps.drop('label', axis=1)
X = X.select_dtypes(exclude='object')  # This excludes all columns of type 'object', typically strings
y = temps['label']  # Keep only the target column



# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize MLPClassifier
# For Healthcare Dataset
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
# For Healthcare Dataset
# Use a tanh activation function because its useful when there is data centered around zero which is true in our case
#3 layers - 7,4,2 neurons in each - 97.8% accuracy
# mlp = MLPClassifier(hidden_layer_sizes=(7, 4, 2), activation='tanh', solver='adam', random_state=1, verbose=True, early_stopping=True, max_iter=300)

# For temp dataset
#2 layers tanh- 2,2 neurons in each - 78.4% accuracy
#2 layers tanh- 4,2 neurons in each - 78.3% accuracy
#3 layers tanh- 16,8,2 neurons in each - 21.2% accuracy
#3 layers relu- 16,8,2 neurons in each - 78.9% accuracy
#3 layers logistic- 8,4,2 - 78.3%
#3 layers logistic- 7,4,2 - 78.3%
#3 layers logistic- 16,8,2 - 78.5%
#3 layers logistic- 9,5,2 - 77.6

mlp = MLPClassifier(hidden_layer_sizes=(3, 2), activation='relu', solver='adam', random_state=1, verbose=True, early_stopping=True, max_iter=300)

# activation=relu, alpha=0.001, hidden_layer_sizes=(8, 4, 2), learning_rate_init=0.1;, score=0.829 total time=   0.2s
# activation=relu, alpha=0.001, hidden_layer_sizes=(9, 5, 2), learning_rate_init=0.1;, score=0.868 total time=   0.4s
# activation=relu, alpha=0.001, hidden_layer_sizes=(9, 5, 2), learning_rate_init=0.1;, score=0.848 total time=   0.2s



# Implementing Grid Search for hyperparameter tuning

'''
param_grid = {
    'hidden_layer_sizes': [(8, 4, 2), (9, 5, 2), (7, 4, 2)],  # Experimenting with different sizes
    'activation': ['tanh', 'relu', 'logistic'],  # Experimenting with different activation functions
    'alpha': [0.0001, 0.001, 0.01],  # Different values for L2 regularization
    'learning_rate_init': [0.001, 0.01, 0.1]  # Different initial learning rates
}

grid_search = GridSearchCV(mlp, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)

# Train the model with the best parameters found by GridSearchCV
best_mlp = grid_search.best_estimator_
best_mlp.fit(X_train, y_train)

#activation=tanh, alpha=0.0001, hidden_layer_sizes=(7, 4, 2), learning_rate_init=0.001;, score=0.966 total time=   2.3s
#activation=tanh, alpha=0.0001, hidden_layer_sizes=(15, 5, 3), learning_rate_init=0.001;, score=0.965 total time=   2.6s
#activation=tanh, alpha=0.001, hidden_layer_sizes=(16, 8, 2), learning_rate_init=0.001;, score=0.968 total time=   4.0s
#activation=tanh, alpha=0.001, hidden_layer_sizes=(7, 4, 2), learning_rate_init=0.001;, score=0.966 total time=   2.7s
#activation=tanh, alpha=0.001, hidden_layer_sizes=(15, 5, 3), learning_rate_init=0.001;, score=0.966 total time=   5.0s
#activation=tanh, alpha=0.01, hidden_layer_sizes=(7, 4, 2), learning_rate_init=0.001;, score=0.978 total time=  10.4s
#activation=tanh, alpha=0.01, hidden_layer_sizes=(7, 4, 2), learning_rate_init=0.001;, score=0.968 total time=   5.5s
#activation=tanh, alpha=0.001, hidden_layer_sizes=(16, 8, 2), learning_rate_init=0.001;, score=0.968 total time=   4.0s
'''

# Re-train using the training data
mlp.fit(X_train, y_train)  # Use X_train and y_train here

# Predict the labels for the test set
predictions = mlp.predict(X_test)

#predictions = best_mlp.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)
