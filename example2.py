import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the stock data into a Pandas DataFrame
df = pd.read_csv('stock_data.csv')

# Split the data into features (X) and target (y)
X = df[['Open', 'Close', 'Volume']]
y = df['Adj Close']

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Use the model to make predictions for the future
future_predictions = model.predict([[100, 105, 200000]])
print(f'Predicted stock price: {future_predictions[0]:.2f}')
