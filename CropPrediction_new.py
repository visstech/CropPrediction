import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Dataset
df = pd.read_csv("c:\ML\FAOSTAT_data.csv")

# Display basic info
print("Dataset Info:")
df.info()
print("\nFirst few rows:")
print(df.head())

# 1. Handle Missing Data
# Check for missing values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Drop rows/columns with excessive missing values (if necessary)
df = df.dropna(thresh=len(df) * 0.5, axis=1)  # Drop columns with more than 50% missing

# Impute missing values for numerical columns
df.fillna(df.mean(), inplace=True)

# 2. Standardize Units and Formats
# Convert necessary columns to numerical types (e.g., Year, Value columns)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# Convert area units if needed (e.g., hectares, tons)
# Add your unit conversion logic if necessary

# 3. Filter Relevant Columns
# Keep columns required for analysis: Area, Year, Item, Element, Value
df = df[['Area', 'Year', 'Item', 'Element', 'Value', 'Unit']]

# Pivot data to make it more structured (optional)
df = df.pivot_table(index=['Area', 'Year', 'Item'],
                    columns='Element',
                    values='Value',
                    aggfunc='sum').reset_index()
df.columns.name = None  # Remove multi-index

# Rename columns for clarity
df.rename(columns={"Area harvested": "Area_Harvested",
                    "Yield": "Yield",
                    "Production": "Production"}, inplace=True)

# 4. Handle Categorical Variables
# Encode categorical variables like Area and Item
label_encoder = LabelEncoder()
df['Area'] = label_encoder.fit_transform(df['Area'])
df['Item'] = label_encoder.fit_transform(df['Item'])

# 5. Remove Outliers
# Use IQR to detect outliers for numerical features like Yield, Production
for col in ['Yield', 'Production', 'Area_Harvested']:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# 6. Normalize or Scale Data
scaler = StandardScaler()
df[['Area_Harvested', 'Yield', 'Production']] = scaler.fit_transform(df[['Area_Harvested', 'Yield', 'Production']])

# Final Cleaned Dataset
print("\nCleaned Dataset:")
print(df.head())

# Save the cleaned dataset
print('Data saved1')
df.to_csv("c:\ML\Cleaned_FAOSTAT_data.csv", index=False)

print('Data saved2')

