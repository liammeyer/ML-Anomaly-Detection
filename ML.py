import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt

# Load Data
train_temps = pd.read_csv('GlobalLandTemperaturesByMajorCity.csv', low_memory=False)
temps = pd.read_csv('MLTempDataset.csv', low_memory=False)

# Preprocess and handle missing data if needed
train_temps.replace([np.inf, -np.inf], np.nan, inplace=True)
train_temps.dropna(inplace=True)
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

train_temps['dt'] = pd.to_datetime(train_temps['dt'])
train_temps['Year'] = train_temps['dt'].dt.year
train_temps['Month'] = train_temps['dt'].dt.month
train_temps['Day'] = train_temps['dt'].dt.day

# Drop unnecessary columns
temps.drop(columns=['index', 'Datetime1', 'Datetime'], inplace=True)
train_temps.drop(columns=['dt', 'AverageTemperatureUncertainty', 'City', 'Country', 'Latitude', 'Longitude'], inplace=True)

# Prepare Data
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

# Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42, stratify=y)

'''
# Define the LSTM model
model = Sequential()
model.add(LSTM(units=75, return_sequences=True, input_shape=(X_reshaped_train.shape[1], X_reshaped_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units=35, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Fit the model
history = model.fit(X_reshaped_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_reshaped_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
'''

def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32), 
                   return_sequences=True, input_shape=(X_reshaped_train.shape[1], X_reshaped_train.shape[2])))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units2', min_value=32, max_value=256, step=32)))
    model.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='intro_to_kt')

tuner.search(X_reshaped_train, y_train, epochs=50, validation_split=0.2)
best_model = tuner.get_best_models()[0]
best_model.evaluate(X_reshaped_test, y_test)
