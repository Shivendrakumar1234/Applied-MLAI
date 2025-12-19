import pandas as pd

# Load CSV file
df = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\HomePrices.csv file.csv")
df.head()

# Create dummy variables for 'town'
dummies = pd.get_dummies(df['town'])
dummies

# Merge dummy variables with original dataframe
merged = pd.concat([df, dummies], axis='columns')
merged

# Drop original 'town' column
final = merged.drop(['town'], axis='columns')
final

# Avoid dummy variable trap (drop one dummy column)
final = final.drop(['west windsor'], axis='columns')
final

# Define X (features) and y (target)
X = final.drop('price', axis='columns')
X

y = final['price']
y

# Train Linear Regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X, y)

# Predictions
model.predict(X)                    # All predictions
model.predict([[2800, 0, 1]])        # 2800 sq ft home in robbinsville
