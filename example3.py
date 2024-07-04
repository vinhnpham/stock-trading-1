#pip install numpy pandas matplotlib scikit-learn

import pandas as pd

# Load the stock data
df = pd.read_csv('stock_data.csv')

# Print the head of the data frame
print(df.head())

# Print the summary statistics of the data frame
print(df.describe())

import datetime

# Convert the date strings to datetime objects
df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))

# Set the date column as the index of the dataframe
df.set_index('date', inplace=True)

# Split the data into training and testing sets
train_data = df[df.index < '2019-01-01']
test_data = df[df.index >= '2019-01-01']

from sklearn.linear_model import LinearRegression

def create_and_train_model(data):
    # Create the linear regression model
    model = LinearRegression()
    
    # Extract the X and y data
    X = data[['open', 'high', 'low', 'close', 'volume']]
    y = data['adj_close']
    
    # Fit the model to the training data
    model.fit(X, y)
    
    return model

# Create and train the model
model = create_and_train_model(train_data)

# Make predictions on the test data
predictions = model.predict(test_data[['open', 'high', 'low', 'close', 'volume']])

# Calculate the mean squared error of the predictions
mse = ((predictions - test_data['adj_close']) ** 2).mean()

# Print the mean squared error
print('Mean Squared Error:', mse)
