# Linear Regression — Review Questions Cheat Sheet

This cheat sheet summarizes the graded review questions you provided, shows the correct answers, and gives brief explanations and small code snippets where helpful.

---

## Question 1
Let X be a dataframe with 100 rows and 5 columns. Let y be the target with 100 samples. Assuming all relevant libraries and data have been imported, the following code has been executed:

```python
LR = LinearRegression()
LR.fit(X, y)
yhat = LR.predict(X)
```

How many samples does `yhat` contain?

- Options: 5 / 500 / **100** / 0

Answer: **100**

Explanation: `LR.predict(X)` returns one prediction per row of `X`. Since `X` has 100 rows, `yhat` (the predicted vector) has 100 samples. The number of columns (features) does not change the number of predictions.

---

## Question 2
What value of R² (coefficient of determination) indicates your model performs best?

- Options: -100 / -1 / 0 / **1**

Answer: **1**

Explanation: R² ranges from -∞ to 1. An R² of 1 means the model explains 100% of the variance in the target (perfect fit). Values near 0 indicate poor explanatory power, and negative values mean the model performs worse than predicting the mean.

Quick code to compute:
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

---

## Question 3
Which statement is true about polynomial linear regression?

- Options:
  - Polynomial linear regression is not linear in any way.
  - **Although the predictor variables of polynomial linear regression are not linear, the relationship between the parameters or coefficients is linear.**
  - Polynomial linear regression uses wavelets.

Answer: **Although the predictor variables of polynomial linear regression are not linear, the relationship between the parameters or coefficients is linear.**

Explanation: Polynomial regression projects input features into polynomial basis functions (e.g., x, x², x³, ...). After transformation, you still fit a linear model (linear in the coefficients). So it's "linear" in parameters but uses nonlinear transformations of predictors.

Example pipeline:
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(X, y)
```

---

## Question 4
The larger the mean squared error (MSE), the better your model performs:

- Options: False / True

Answer: **False**

Explanation: Mean Squared Error (MSE) measures average squared difference between true and predicted values:
MSE = (1/n) * Σ (y_i - ŷ_i)².
Lower MSE indicates better predictive accuracy. A larger MSE means worse performance.

Quick code:
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

---

## Question 5
Assume libraries are imported. y is the target and X is the features. Consider:

```python
Input = [('scale', StandardScaler()), ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(X, y)
ypipe = pipe.predict(X)
```

What is the result of `ypipe`?

- Options:
  - Polynomial transform, standardize the data, then perform a prediction using a linear regression model.
  - **Standardize the data, then perform prediction using a linear regression model.**
  - Polynomial transform, then standardize the data.

Answer: **Standardize the data, then perform prediction using a linear regression model.**

Explanation: Pipeline steps are executed in order. `StandardScaler()` scales features first; then `LinearRegression()` is fitted on the scaled features. Predictions are produced by passing X through the scaler and then the model.

---

## Quick Reference — Important Points
- predict(X) → returns one prediction per row of X.
- R² = 1 is best; larger is better up to 1.
- Polynomial regression = nonlinear features but linear in coefficients.
- Lower MSE is better.
- Pipeline steps run in sequence: transform(s) → estimator.

---

## Small examples (put into your Python REPL)
1. Predictions length:
```python
yhat = LR.predict(X)
len(yhat)  # equals X.shape[0]
```

2. R² and MSE:
```python
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y, yhat)
mse = mean_squared_error(y, yhat)
```

---

