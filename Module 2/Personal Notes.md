# 🧹 Data Cleaning & Preparation in Python (Pandas) 🐼

This document summarizes key techniques for cleaning and preparing data for analysis using the Python Pandas library.

---

## Dealing with Missing Data ❓

Missing data is common. Here are the main strategies:

1. **Check the Source**: Try to find the missing information from where the data was collected.  
2. **Drop Missing Values**: Remove rows or columns with missing data.
   - Drop the entire variable/column (if it has too many missing values or isn't important).
   - Drop the specific data entry/row (often the best if only a few rows are missing).
3. **Replace Missing Values**: Fill in the gaps with estimated values.
   - Replace with the **average** (mean) of the column (or similar data points).
   - Replace with the **most frequent** value (mode).
   - Replace based on **other functions** or more complex imputation methods.
4. **Leave as Missing**: Sometimes, depending on the analysis, it's okay to leave them as is.

---

### Dropping Missing Values in Pandas

Use the `.dropna()` method.

```python
# Drop rows with any missing values
df_cleaned = dataframes.dropna()

# Drop rows where a specific column ('column_name') has missing values
# axis=0 specifies rows
df_cleaned = dataframes.dropna(subset=['column_name'], axis=0)

# Drop columns with any missing values
# axis=1 specifies columns
df_cleaned = dataframes.dropna(axis=1)

# To make the changes directly to the original DataFrame, use inplace=True
dataframes.dropna(subset=['column_name'], axis=0, inplace=True)
```

Key Point: Remember `inplace=True` modifies the DataFrame directly. Without it, `.dropna()` returns a new DataFrame, leaving the original unchanged.

---

## Replacing Missing Values in Pandas

Use the `.replace()` method or `.fillna()`.

```python
import numpy as np  # Often missing values are represented as np.nan

# Method 1: Using .replace()
# Replace NaN in 'column_name' with the column's mean
mean_value = df['column_name'].mean()
df['column_name'].replace(np.nan, mean_value, inplace=True)

# Method 2: Using .fillna() (often preferred for NaN)
# Replace NaN in 'column_name' with the column's mean
mean_value = df['column_name'].mean()
df['column_name'].fillna(mean_value, inplace=True)

# Replace NaN with the most frequent value (mode)
mode_value = df['column_name'].mode()[0]  # .mode() returns a Series, take the first element
df['column_name'].fillna(mode_value, inplace=True)
```

---

## Data Formatting 📐

Data often comes in inconsistent formats. Standardizing formats is crucial.

- Example: Standardizing state names (e.g., "NY", "newyork", "N.Y." all become "New York").
- Example: Converting date formats.

```python
import pandas as pd

# Convert a date column from 'MM/DD/YYYY' to Pandas datetime objects
df['date_column'] = pd.to_datetime(df['date_column'], format='%m/%d/%Y')
```

---

## Applying Calculations to Columns

You can easily perform mathematical operations on entire columns.

```python
# Convert 'temp_celsius' to 'temp_fahrenheit'
df['temp_fahrenheit'] = df['temp_celsius'] * 9/5 + 32

# Convert 'city-mpg' (miles per gallon) to 'city-L/100km' (liters per 100km)
df["city-L/100km"] = 235 / df["city-mpg"]

# Rename the column (optional but good practice)
df.rename(columns={"city-mpg": "city-L/100km"}, inplace=True)
```

---

## Correcting Data Types ⚙️

Data might be stored incorrectly (e.g., numbers as text).

Common Pandas Data Types:
- object: Usually strings (text).
- int64: Integer numbers.
- float64: Floating-point (decimal) numbers.
- datetime64: Date and time values.
- bool: True/False values.

Checking and Correcting Types:

```python
# Check data types of all columns
print(df.dtypes)

# Convert a column ('Price') from object/string to float
df['Price'] = df['Price'].astype(float)

# Convert a column ('CustomerID') to string/object
df['CustomerID'] = df['CustomerID'].astype(str)
```

---

## Data Normalization 📏

Normalization scales data into a similar range, typically between 0 and 1, or with a mean of 0 and standard deviation of 1. This is important when features have vastly different scales (like 'Age' vs 'Income').

Normalization Methods:

- Simple Feature Scaling: Divide by the maximum value. Scales data between 0 and 1.

```python
df['column'] = df['column'] / df['column'].max()
```

- Min-Max Scaling: Rescales data to be between 0 and 1.

```python
df['column'] = (df['column'] - df['column'].min()) / (df['column'].max() - df['column'].min())
```

- Z-Score Standardization (Standard Scaling): Rescales data to have a mean of 0 and a standard deviation of 1.

```python
df['column'] = (df['column'] - df['column'].mean()) / df['column'].std()
```

---

## Binning (Discretization) 📊

Binning groups continuous data into discrete intervals ("bins"). This can help reduce the impact of small observation errors and reveal underlying patterns.

Example: Grouping 'price' values into categories like 'Low', 'Medium', 'High'.

```python
import numpy as np
import pandas as pd

# Define bin edges (4 edges create 3 bins)
min_val = df['price'].min()
max_val = df['price'].max()
bins = np.linspace(min_val, max_val, 4)  # Creates 3 equal-width bins

# Define bin labels
group_names = ['Low', 'Medium', 'High']

# Apply binning
df['binned_price'] = pd.cut(
    df['price'],
    bins,
    labels=group_names,
    include_lowest=True  # Include the lowest value in the first bin
)

# Display the value counts for the new bins
print(df['binned_price'].value_counts())
```

---

## Turning Categorical Variables into Quantitative Variables 🔢

Most statistical models require numerical input. Categorical variables (like 'fuel type' with values "Gas" or "Diesel") need to be converted.

One-Hot Encoding is a common technique. It creates new binary (0 or 1) columns for each category.

Example: A 'Car Fuel' column with "Gas" and "Diesel" becomes two columns: 'Gas' and 'Diesel'.

If 'Car Fuel' is "Gas", then 'Gas' = 1 and 'Diesel' = 0.  
If 'Car Fuel' is "Diesel", then 'Gas' = 0 and 'Diesel' = 1.

One-Hot Encoding in Pandas:

```python
import pandas as pd

# Get dummy variables for the 'Car Fuel' column
dummy_variables = pd.get_dummies(df['Car Fuel'])

# Add the new dummy columns back to the original DataFrame
df = pd.concat([df, dummy_variables], axis=1)

# Optionally, drop the original categorical column
df.drop('Car Fuel', axis=1, inplace=True)
```

This transforms your categorical data into a numerical format suitable for modeling.