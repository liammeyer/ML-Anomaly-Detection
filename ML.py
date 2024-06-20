import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt
from sklearn.model_selection import KFold

# Load Data
# train_temps = pd.read_csv('GlobalLandTemperaturesByMajorCity.csv', low_memory=False)
# temps = pd.read_csv('MLTempDataset.csv', low_memory=False)
sensor = pd.read_csv('response.csv', low_memory=False)

# Preprocess and handle missing data if needed
# train_temps.replace([np.inf, -np.inf], np.nan, inplace=True)
# train_temps.dropna(inplace=True)
# temps.replace([np.inf, -np.inf], np.nan, inplace=True)
# temps.dropna(inplace=True)
# sensor.replace([np.inf, -np.inf], np.nan, inplace=True)
# sensor.dropna(inplace=True)

# Feature engineering for datetime column
# Convert object type in Datetime column to feature
'''
temps['Datetime'] = pd.to_datetime(temps['Datetime'])
temps['Year'] = temps['Datetime'].dt.year
temps['Month'] = temps['Datetime'].dt.month
temps['Day'] = temps['Datetime'].dt.day
temps['Hour'] = temps['Datetime'].dt.hour
temps['Minute'] = temps['Datetime'].dt.minute

train_temps['dt'] = pd.to_datetime(train_temps['dt'])
train_temps['Year'] = train_temps['dt'].dt.year
train_temps['Month'] = train_temps['dt'].dt.month
train_temps['Day'] = train_temps['dt'].dt.data
'''


# Drop unnecessary columns
# temps.drop(columns=['index', 'Datetime1', 'Datetime'], inplace=True)
# train_temps.drop(columns=['dt', 'AverageTemperatureUncertainty', 'City', 'Country', 'Latitude', 'Longitude'], inplace=True)

# Prepare Data
'''
X_test = temps.drop('label', axis=1).select_dtypes(exclude='object')
y_test = temps['label']
X_train = train_temps.drop('label', axis=1).select_dtypes(exclude='object')
y_train = train_temps['label']

# Scale features
scaler = StandardScaler()
X_scaled_test = scaler.fit_transform(X_test)
X_scaled_train = scaler.fit_transform(X_test)

# Reshape input to be [samples, time steps, features]
X_reshaped_test = X_scaled_test.reshape((X_scaled_test.shape[0], 1, X_scaled_test.shape[1]))
X_reshaped_train = X_scaled_train.reshape((X_scaled_train.shape[0], 1, X_scaled_train.shape[1]))
'''

X = sensor.drop('sensors__lsid', axis=1)
X = sensor.drop('station_id_uuid', axis=1).select_dtypes(exclude='object')
y = sensor['sensors__lsid']
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=25, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Fit the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")


