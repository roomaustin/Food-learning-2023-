from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
data = pd.read_csv('health_dataset.csv')

# Split the data into inputs and output
X = data.drop('outcome', axis=1)
y = data['outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on the testing data
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Model accuracy: {accuracy}')

# Create a DataFrame for the test data and predictions
test_data = X_test.copy()
test_data['actual_outcome'] = y_test
test_data['predicted_outcome'] = predictions

# Export the data to an Excel file
test_data.to_excel('predictions.xlsx')
