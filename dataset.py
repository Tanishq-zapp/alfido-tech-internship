import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Create a sample dataset
data = {
    'age': [25, 35, 45, 20, 50, np.nan, 30, 40],
    'salary': [50000, 60000, 80000, 40000, 90000, 75000, 55000, 65000],
    'city': ['barcelona', 'London', 'wales', 'barcelona', 'wales', 'London', 'barcelona', 'wales'],
    'experience': [2, 5, 10, 1, 15, 8, 3, 7],
    'department': ['Sales', 'IT', 'Marketing', 'Sales', 'IT', 'Marketing', 'Sales', 'IT']
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Impute missing values in the 'age' column with the median
imputer = SimpleImputer(strategy='median')
df['age'] = imputer.fit_transform(df[['age']])
print("\nDataFrame after imputing missing 'age' values:")
print(df)

# One-hot encode the 'city' and 'department' columns
df = pd.get_dummies(df, columns=['city', 'department'], drop_first=True)
print("\nDataFrame after one-hot encoding categorical features:")
print(df)

# Select the numerical features to scale
numerical_cols = ['age', 'salary', 'experience']
df_to_scale = df[numerical_cols]

# Apply StandardScaler
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df_to_scale)

print("\nFinal DataFrame after standardization:")
print(df)