import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed

# Load Data
file_path = '/ML-Anomaly-Detection/data/response.csv'
sensor_data = pd.read_csv(file_path, low_memory=False)

# Print the first few rows to understand the data structure
print(sensor_data.head())

# Extract relevant columns
# Replace these column names with actual names from your dataset
relevant_columns = ['temp', 'wind_speed', 'barometric_pressure']

# Ensure all relevant columns exist in the dataset
missing_columns = [col for col in relevant_columns if col not in sensor_data.columns]
if missing_columns:
    raise ValueError(f"Columns {missing_columns} not found in the dataset")

sensor_filtered = sensor_data[relevant_columns]

# Preprocess and handle missing data
sensor_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)

# Print the number of missing values in each column
print(sensor_filtered.isnull().sum())

# Fill missing values using median strategy
imputer = SimpleImputer(strategy='median')
sensor_filled = pd.DataFrame(imputer.fit_transform(sensor_filtered), columns=sensor_filtered.columns)

# Verify if there are any remaining NaN values
print(sensor_filled.isnull().sum())

# Extract deciles from 10-second blocks (assuming 10 samples/second)
def extract_deciles(block):
    return np.percentile(block, [10, 20, 30, 40, 50, 60, 70, 80, 90], axis=0)

block_size = 100  # 10 seconds of data
blocks = [extract_deciles(sensor_filled[i:i+block_size]) for i in range(0, len(sensor_filled), block_size) if len(sensor_filled[i:i+block_size]) == block_size]
deciles_data = np.array(blocks)

# Prepare training data (12 days * 24 hours * 60 minutes * 6 blocks/minute)
num_blocks_per_day = 8640
num_days = 12
train_data = deciles_data[:num_blocks_per_day*num_days]

# Prepare LSTM input (10 minutes = 60 blocks, forecast next 5 minutes = 30 blocks)
X = []
y = []
time_steps = 60  # 10 minutes
forecast_steps = 30  # 5 minutes

for i in range(len(train_data) - time_steps - forecast_steps):
    X.append(train_data[i:i+time_steps])
    y.append(train_data[i+time_steps:i+time_steps+forecast_steps])

X = np.array(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(25, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(forecast_steps * deciles_data.shape[2], activation='linear'))
lstm_model.compile(optimizer='adam', loss='mse')

# Train the LSTM model
history = lstm_model.fit(X_train, y_train.reshape(y_train.shape[0], -1), epochs=20, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the LSTM model
lstm_loss = lstm_model.evaluate(X_test, y_test.reshape(y_test.shape[0], -1))
print(f"LSTM Test Loss: {lstm_loss:.4f}")

# Autoencoder model for feature extraction
input_layer = Input(shape=(time_steps, deciles_data.shape[2]))
encoded = LSTM(128, activation='relu', return_sequences=True)(input_layer)
encoded = LSTM(64, activation='relu', return_sequences=False)(encoded)
decoded = RepeatVector(time_steps)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(deciles_data.shape[2]))(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the Autoencoder model
history = autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.2, verbose=2)

# Feature extraction
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-5].output)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# One-Class SVM for anomaly detection
oc_svm = OneClassSVM(gamma='auto')
oc_svm.fit(X_train_encoded.reshape(X_train_encoded.shape[0], -1))

# Predict and evaluate
y_pred_train = oc_svm.predict(X_train_encoded.reshape(X_train_encoded.shape[0], -1))
y_pred_test = oc_svm.predict(X_test_encoded.reshape(X_test_encoded.shape[0], -1))

print("Training data predictions:", y_pred_train)
print("Testing data predictions:", y_pred_test)
