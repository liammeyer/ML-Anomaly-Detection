import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load Data
sensor = pd.read_csv('response.csv', low_memory=False)

# Preprocess and handle missing data
sensor.replace([np.inf, -np.inf], np.nan, inplace=True)
sensor.dropna(subset=['sensors__lsid'], inplace=True)

# Drop columns with excessive missing values (threshold set at 50%)
threshold = 0.5
sensor_filtered = sensor.loc[:, sensor.isnull().mean() < threshold]

# Fill remaining missing values with median
imputer = SimpleImputer(strategy='median')
sensor_filled = pd.DataFrame(imputer.fit_transform(sensor_filtered), columns=sensor_filtered.columns)

# Prepare Data
X = sensor_filled.drop(['sensors__lsid', 'station_id_uuid'], axis=1, errors='ignore')
y = sensor_filled['sensors__lsid']

# Train/Test Split without stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input to be [samples, time steps, features] for LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=25, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
