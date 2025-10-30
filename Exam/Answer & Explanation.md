# Comprehensive Answers with Explanations

## 1) What does `df.dropna(subset=["price"], axis=0)` do?
- **Answer:** Drop the “not a number” values from the column "price".  
- **Explanation:** `dropna` with `subset=["price"]` removes rows where the specified column has `NaN`. `axis=0` refers to dropping rows.

---

## 2) How to provide many summary statistics for all columns in `df`?
- **Answer:** `df.describe(include="all")`  
- **Explanation:** `describe(include="all")` computes summary statistics for both numeric and non-numeric columns (counts, unique, top, freq for categorical; mean, std, etc. for numeric).

---

## 3) How to find the shape of dataframe `df`?
- **Answer:** `df.shape`  
- **Explanation:** `shape` returns a tuple `(n_rows, n_columns)`.

---

## 4) What does `df.to_csv("A.csv")` do?
- **Answer:** Save the dataframe `df` to a csv file called `"A.csv"`.  
- **Explanation:** `to_csv` writes the DataFrame to a CSV file at the given filename/path.

---

## 5) What does `result = np.linspace(min(df["city-mpg"]), max(df["city-mpg"]), 5)` do?
- **Answer:** Builds a bin array ranging from the smallest value to the largest value of "city-mpg" in order to build 4 bins of equal length.  
- **Explanation:** Explanation: `np.linspace(start, stop, num=5)` returns 5 evenly spaced values including the endpoints. The number of intervals between those boundary points is `num - 1`, so 5 points → 4`.

---

## 6) What does `df['peak-rpm'].replace(np.nan, 5, inplace=True)` do?
- **Answer:** Replace the `NaN` values with `5` in the column `'peak-rpm'`.  
- **Explanation:** `replace` with `np.nan` as the target and `inplace=True` modifies the column in-place, filling missing values with `5`.

---

## 7) How to one-hot encode the column `'fuel-type'` in `df`?
- **Answer:** `pd.get_dummies(df["fuel-type"])`  
- **Explanation:** `get_dummies` produces dummy/indicator columns for each category. Often you’ll use `pd.get_dummies(df, columns=["fuel-type"])` to add encoded columns back to the DataFrame.

---

## 8) What does the vertical axis (y-axis) on a scatterplot represent?
- **Answer:** Dependent variable.  
- **Explanation:** Conventionally, the y-axis represents the response or dependent variable; the x-axis is the predictor.

---

## 9) What does the horizontal axis (x-axis) on a scatterplot represent?
- **Answer:** Independent variable.  
- **Explanation:** The x-axis usually shows the predictor/independent variable.

---

## 10) If we have 10 columns and 100 samples, how large is the output of `df.corr()`?
- **Answer:** 10 × 10  
- **Explanation:** `corr()` returns a square correlation matrix across columns (features), not samples. With 10 columns, the result is a 10×10 matrix.

---

## 11) What is the largest possible element resulting from `df.corr()`?
- **Answer:** `1`  
- **Explanation:** Pearson correlation coefficients range from `-1` to `+1`; `+1` indicates a perfect positive linear relationship.

---

## 12) If the Pearson correlation of two variables is zero:
- **Answer:** The two variables are not correlated (specifically, there is no *linear* correlation).  
- **Explanation:** Zero Pearson correlation indicates no linear association; there may still be non-linear relationships.

---

## 13) If the p-value of the Pearson correlation is `1`:
- **Answer:** The variables are not correlated.  
- **Explanation:** A p-value of 1 provides no evidence against the null hypothesis of zero correlation; it indicates the observed correlation is not statistically significant.

---

## 14) What does `lm = LinearRegression()` do?
- **Answer:** Create a linear regression object.  
- **Explanation:** This instantiates the estimator; you later fit it with `lm.fit(X, y)` to train.

---

## 15) If the predicted function is `Yhat = a + b1 X1 + b2 X2 + b3 X3 + b4 X4`, the method is:
- **Answer:** Multiple Linear Regression.  
- **Explanation:** A linear model with multiple predictors is multiple linear regression.

---

## 16) What do the pipeline lines perform?
Code:
```python
Input = [('scale', StandardScaler()), ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(Z, y)
ypipe = pipe.predict(Z)
```
- **Answer:** Standardize the data, then perform a prediction using a linear regression model with features `Z` and targets `y`.  
- **Explanation:** The pipeline applies `StandardScaler()` to `Z`, fits `LinearRegression()` on the scaled features, then uses the pipeline to predict.

---

## 17) What is the maximum value of R² that can be obtained?
- **Answer:** `1`  
- **Explanation:** R² ranges up to `1`; `1` indicates a perfect fit on the evaluated data.

---

## 18) `PolynomialFeatures(degree=2)` — what is the order of the polynomial?
- **Answer:** `2`  
- **Explanation:** Degree 2 produces quadratic terms and pairwise interaction terms (and usually includes lower-degree terms when `include_bias` and `include_interaction` defaults are used).

---

## 19) You have a linear model with average training R² = 0.5. After a 100th-order polynomial transform, training R² = 0.99. Which comment is correct?
- **Answer:** The results on your training data are not the best indicator of how your model performs. You should use your test data to get a better idea.  
- **Explanation:** A huge jump in training performance after a very high-degree transform likely indicates overfitting. Use a held-out test set or cross-validation to evaluate generalization.

---

## 20) Ridge regression: `R²_val = 1` and `R²_train = 0.5`. What should you do?
- **Answer / Guidance:** Investigate data leakage and validation procedure; the low training R² suggests underfitting but a perfect validation R² is suspicious. Do not blindly increase the regularization parameter `alpha`. Instead:
  - Verify the validation set is truly held out (no leakage, no repeated tuning on validation).
  - Use proper cross-validation to tune `alpha`.
  - If the model is truly underfitting, consider decreasing `alpha` (less regularization) or increasing model capacity (e.g., features or polynomial terms), then evaluate with a properly held-out test set.
- **Explanation:** A perfect validation score when training performance is low usually indicates an error in the validation process (data leakage) or repeated tuning on the same validation set. Increasing `alpha` (more regularization) would typically *worsen* a low training R², not help it.

---