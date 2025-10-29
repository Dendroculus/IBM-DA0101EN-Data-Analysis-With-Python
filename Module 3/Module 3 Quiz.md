# 📚 IBM DA0101EN - Graded Review Questions: Answers & Explanations (Module 3) 🐍

*This README contains the correct answers and short explanations for the Graded Review Questions from Module 3. Copy this file into your repository (README.md) and push to GitHub.*

---

## ❓ Question 1
Consider the dataframe "df". What method provides the summary statistics?

- ✅ **df.describe()**

**Explanation:**  
`df.describe()` returns summary statistics for numeric columns (count, mean, std, min, 25%, 50%, 75%, max). It’s the standard Pandas method for quick descriptive statistics.

---

## ❓ Question 2
Consider the following dataframe:

df_test = df['body-style', 'price']

The following operation is applied:

df_grp = df_test.groupby(['body-style'], as_index=False).mean()

What are resulting values of df_grp['price']?

- ✅ **The average price for each body style.**

**Explanation:**  
Grouping by `'body-style'` and calling `.mean()` computes the mean of numeric columns (here `price`) for each group. So `df_grp['price']` contains the average price for each unique body style.

---

## ❓ Question 3
Correlation implies causation:

- ✅ **False**

**Explanation:**  
Correlation measures association between two variables but does not prove one causes the other. Confounders or coincidence can produce correlations without causality.

---

## ❓ Question 4
What is the minimum possible value of Pearson's Correlation?

- ✅ **-1**

**Explanation:**  
Pearson's correlation coefficient ranges from -1 to 1. -1 indicates a perfect negative linear relationship.

---

## ❓ Question 5
What is the Pearson correlation between variables X and Y if X = Y?

- ✅ **1**

**Explanation:**  
If X equals Y exactly, there is a perfect positive linear relationship; Pearson's r = 1.

---

## Quick reminders
- `df.describe()` — summary statistics for numeric data (use `include='all'` to include non-numeric).
- `groupby(...).mean()` — computes the mean of numeric columns per group.
- Correlation ≠ causation — correlations can be caused by confounders, bias, or chance.
- Pearson r ∈ [-1, 1]; 1 = perfect positive, -1 = perfect negative, 0 ≈ no linear relationship.

---

<p align="center">
  <em>Hope this helps!</em> 
</p>