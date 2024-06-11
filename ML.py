import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load Data
temps = pd.read_csv('MLTempDataset.csv', low_memory=False)

# Preprocess and handle missing data if needed
temps.replace([np.inf, -np.inf], np.nan, inplace=True)
temps.dropna(inplace=True)

# Feature engineering for datetime column
# Convert object type in Datetime column to feature
temps['Datetime'] = pd.to_datetime(temps['Datetime'])
temps['Year'] = temps['Datetime'].dt.year
temps['Month'] = temps['Datetime'].dt.month
temps['Day'] = temps['Datetime'].dt.day
temps['Hour'] = temps['Datetime'].dt.hour
temps['Minute'] = temps['Datetime'].dt.minute

# Drop unnecessary columns
temps.drop(columns=['index', 'Datetime1', 'Datetime'], inplace=True)

# Prepare Data
X = temps.drop('label', axis=1).select_dtypes(exclude='object')
y = temps['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input to be [samples, time steps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42, stratify=y)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

print(classification_report(y_test, y_pred))
