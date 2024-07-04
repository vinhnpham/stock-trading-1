from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the stock data
df = pd.read_csv("stock_data.csv")

# Split the data into features (X) and target (y)
X = df[["Previous Day Price", "Volume"]]
y = df["Price"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Print the predictions
print(predictions)
