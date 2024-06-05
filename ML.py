
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Load Data
data = pd.read_csv('patientMonitoring.csv')

data.replace([np.inf, -np.inf], np.nan, inplace=True) #replace all pos inf and neg inf with nan (not any number)

#Converting all to numberic
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Convert non-numeric columns to categorical or datetime, as appropriate
non_numeric_columns = data.select_dtypes(exclude=['int64', 'float64']).columns
for column in non_numeric_columns:
    #if column == 'Timestamp':
        #data[column] = pd.to_datetime(data[column], format='%d/%m/%Y %H:%M').astype(int)
    data[column] = data[column].astype('category')

data = data.dropna() #REMOVES ALL ROWS containing 'not any numberic' values - probably N/A but just in case

data = pd.get_dummies(data, drop_first=True)

print ("made it to here1")

# Step 2: Prepare Data
X = data.drop('label', axis=1)  # Assuming 'target_column_name' is the name of your target column
y = data['label']

print ("made it to here2")

# Step 3: Train/Test Split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
