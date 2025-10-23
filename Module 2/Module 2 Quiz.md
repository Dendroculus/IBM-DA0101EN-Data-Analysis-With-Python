# 📚 IBM DA0101EN - Graded Review Questions: Answers & Explanations 🐍

*This README contains the correct answers and short explanations for the Graded Review Questions shown in your submission. Copy this file into your repository (README.md) and push to GitHub.*

---

## ❓ Question 1
Consider the dataframe `df`. What is the result of the following operation: `df['symbolling'] = df['symbolling'] + 1`?

- ✅ **Every element in the column "symbolling" will increase by one.**

**Explanation:**  
`df['symbolling']` selects the column (a Pandas Series). Adding `+ 1` performs element-wise addition, increasing each value in that column by 1. The assignment stores the updated Series back into `df['symbolling']`.

---

## ❓ Question 2
Consider the dataframe `df`. What does the command `df.rename(columns={'a':'b'})` change about the dataframe `df`?

- ✅ **Renames column "a" of the dataframe to "b".**

**Explanation:**  
`df.rename(columns={'a':'b'})` returns a new DataFrame with the column named `'a'` renamed to `'b'`. By default it does not modify `df` in place. To keep the change you can either assign it back (`df = df.rename(columns={'a':'b'})`) or use `df.rename(columns={'a':'b'}, inplace=True)`.

---

## ❓ Question 3
Consider the dataframe `df`. What is the result of the following operation `df['price'] = df['price'].astype(int)`?

- ✅ **Convert or cast the column 'price' to an integer value.**

**Explanation:**  
`.astype(int)` applied to a Series converts that Series to integer dtype (if possible). The assignment `df['price'] = ...` overwrites the `price` column with the integer-casted values.

---

## ❓ Question 4
Consider the column of the dataframe `df['a']`. The column has been standardized. What is the standard deviation of the values as a result of applying the following operation: `df['a'].std()`?

- ✅ **1**

**Explanation:**  
Standardization typically means transforming data to z-scores via `(x - mean) / std`. After this transformation the column's mean ≈ 0 and standard deviation ≈ 1 (depending on sample vs population definitions). So `df['a'].std()` should return approximately `1`.

---

## ❓ Question 5a
Consider the column `df['Fuel']` with two values: `'gas'` and `'diesel'`. What will be the name of the new columns from `pd.get_dummies(df['Fuel'])`?

- ✅ **'gas' and 'diesel'**

**Explanation:**  
`pd.get_dummies()` creates one column per unique category. For categories `'gas'` and `'diesel'`, the resulting DataFrame will have columns named `'gas'` and `'diesel'` (unless you supply a prefix or rename them).

---

## ❓ Question 5b
What are the values of the new columns from part 5a?

- ✅ **1 and 0**

**Explanation:**  
Dummy (indicator) columns contain binary values: `1` indicates the row's category is present, `0` means it is not. For a row with Fuel `'gas'`, column `'gas'` will be `1` and `'diesel'` will be `0`.

---

## Quick pandas reminders
- `df.rename(columns={'a':'b'})` — returns a DataFrame with column `'a'` renamed to `'b'`. Use `inplace=True` or reassign to `df` to keep changes.
- `df['col'].astype(type)` — cast a single column to a different dtype.
- Standardization → `(x - mean) / std` → resulting std ≈ 1, mean ≈ 0.
- `pd.get_dummies(series)` → creates 0/1 indicator columns, one per category.

---

<p align="center">
  <em>Hope this helps!</em> 🎉
</p>