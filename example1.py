import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the stock data
df = pd.read_csv('stock_data.csv')

# Select the features to use for prediction
X = df[['Open', 'Close', 'Volume']]

# Select the target variable
y = df['Price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the model and fit it to the training data
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Use the model to make predictions on the test data
predictions = model.predict(X_test)

# Calculate the mean absolute error between the predictions and the true values
mae = np.mean(abs(predictions - y_test))

print(f'Mean Absolute Error {mae}')
