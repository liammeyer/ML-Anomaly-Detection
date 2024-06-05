
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Load Data
attack = pd.read_csv('Attack.csv')
patient = pd.read_csv('patientMonitoring.csv')
environment = pd.read_csv('environmentMonitoring.csv')

attack.replace([np.inf, -np.inf], np.nan, inplace=True) #replace all pos inf and neg inf with nan (not any number)
patient.replace([np.inf, -np.inf], np.nan, inplace=True) #replace all pos inf and neg inf with nan (not any number)
environment.replace([np.inf, -np.inf], np.nan, inplace=True) #replace all pos inf and neg inf with nan (not any number)

attack = attack.dropna() #REMOVES ALL ROWS containing 'not any numberic' values - probably N/A but just in case
patient = patient.dropna() #REMOVES ALL ROWS containing 'not any numberic' values - probably N/A but just in case
environment = environment.dropna() #REMOVES ALL ROWS containing 'not any numberic' values - probably N/A but just in case


#Prepare Data
X = data.drop('label', axis=1)  # Assuming 'target_column_name' is the name of your target column
y = data['label']

# Step 3: Train/Test Split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
