import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the data
file_path = 'response.csv'  # Update this with the path to your file
sensor_data = pd.read_csv(file_path, low_memory=False)

# Display the first few rows and column names to understand the data structure
print(sensor_data.head())
print(sensor_data.columns.tolist())

# Identify the relevant columns for temperature, wind speed, and barometric pressure
relevant_columns = ['sensors__data__temp_out', 'sensors__data__wind_speed_avg', 'sensors__data__wind_speed_hi', 'sensors__data__wind_dir_of_hi', 'sensors__data__pressure_last', 'sensors__lsid']

# Check if the relevant columns are in the dataset
for col in relevant_columns:
    if col not in sensor_data.columns:
        raise ValueError(f"Column '{col}' not found in the dataset")

# Filter relevant columns
sensor_filtered = sensor_data[relevant_columns]

# Preprocess and handle missing data
sensor_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)
sensor_filtered.dropna(subset=['sensors__lsid'], inplace=True)

# Fill remaining missing values with median
imputer = SimpleImputer(strategy='median')
sensor_filled = pd.DataFrame(imputer.fit_transform(sensor_filtered), columns=sensor_filtered.columns)

# Prepare Data
X = sensor_filled.drop('sensors__lsid', axis=1)
y = sensor_filled['sensors__lsid']

# Train/Test Split without stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input to be [samples, time steps, features]
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=25, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))


'''
# Define a simple Dense model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid for binary classification
'''


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=4, validation_split=0.2, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
