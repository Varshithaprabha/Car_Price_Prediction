import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
df = pd.read_csv('car_data.csv')  # Replace with your actual file path

# Step 2: Check and clean column names (strip any spaces)
df.columns = df.columns.str.strip()

# Step 3: Feature and Target Selection
X = df.drop(['Car_Name', 'Selling_Price'], axis=1)  # Drop 'Car_Name' and 'Selling_Price'
y = df['Selling_Price']  # Target variable (Selling_Price)

# Step 4: Encoding categorical variables
le = LabelEncoder()

# Ensure to encode only the categorical columns: Fuel_Type, Selling_type, and Transmission
X['Fuel_Type'] = le.fit_transform(X['Fuel_Type'])
X['Selling_type'] = le.fit_transform(X['Selling_type'])
X['Transmission'] = le.fit_transform(X['Transmission'])

# Step 5: Check for missing values and handle them (if any)
if X.isnull().sum().any() or y.isnull().sum() > 0:
    # Fill missing values (using mean for numerical columns)
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Output the evaluation results
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-Squared: {r2}')
