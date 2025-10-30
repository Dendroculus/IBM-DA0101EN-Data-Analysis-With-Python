# Regression: Comprehensive Notes

These notes summarize Simple Linear Regression (SLR), Multiple Linear Regression (MLR), polynomial regression, preprocessing & pipelines, visualization for model evaluation, and numerical evaluation metrics (MSE and R²). Included are conceptual explanations, common pitfalls, interpretation guidance, and practical Python examples using scikit-learn, pandas, seaborn, and numpy.

---

## 1. Core Concepts

- Regression: Predict a continuous target variable y from one or more predictor (independent) variables X.
- Simple Linear Regression (SLR): One predictor x, one target y. Model form:
  ŷ = b0 + b1 * x
  - b0: intercept (value of ŷ when x = 0)
  - b1: slope (change in ŷ per unit change in x)
- Multiple Linear Regression (MLR): Two or more predictors:
  ŷ = b0 + b1*x1 + b2*x2 + ... + bk*xk

Notes:
- ŷ denotes a predicted/estimated value.
- Training (fitting) the model finds the parameters (b0, b1, ...).
- Observations deviate from the model due to noise and model mismatch.

---

## 2. Training and Prediction (scikit-learn)

Typical steps:
1. Prepare features X and target y (pandas DataFrame / numpy arrays). Each row = a sample.
2. Create `LinearRegression()` object.
3. Fit: `lm.fit(X, y)` → learns intercept and coefficients.
4. Predict: `y_pred = lm.predict(X_new)`.
5. Inspect parameters: `lm.intercept_`, `lm.coef_`.

Example (SLR):
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Example data
df = pd.DataFrame({
    'highway_mpg': [15, 18, 22, 25, 30],
    'price': [30000, 28000, 22000, 20000, 13771.30]
})

X = df[['highway_mpg']]  # must be 2D
y = df['price']

lm = LinearRegression()
lm.fit(X, y)

print("Intercept (b0):", lm.intercept_)
print("Slope (b1):", lm.coef_[0])

# Predict price for highway_mpg = 20
print("Predicted price for 20 mpg:", lm.predict([[20]])[0])
```

Interpretation:
- b1 tells how much price changes as highway_mpg increases by one unit.
- If slope is negative, price decreases when mpg increases.

Caveats:
- Predictions outside the range of training data (extrapolation) may be unrealistic (e.g., negative prices).
- Model assumes linear relationship between predictors and the target.

---

## 3. Visual Evaluation

Visualization is the first tool to check whether model results make sense.

1. Regression plot
   - Shows scatter of data and fitted regression line.
   - Useful to see direction (positive/negative) and apparent linearity.

2. Residual plot
   - Residual = y - ŷ (or ŷ - y; be consistent).
   - Plot residuals (vertical) vs predictor (horizontal).
   - Good linear fit: residuals randomly scattered around 0 with roughly constant variance.
   - Bad signs:
     - Curvature → non-linear relationship (consider polynomial features or other models).
     - Patterned residuals → model missing structure.
     - Funnel shape (variance increases/decreases with x) → heteroscedasticity.

3. Distribution plot (predicted vs actual)
   - Plots estimated distribution of predicted values and true target values.
   - Useful for MLR and higher-dimensional cases where single scatter plots are not practical.
   - Overlap suggests predictions match targets; diverging regions indicate model weaknesses.

Examples (seaborn):
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Regression plot (SLR)
sns.regplot(x='highway_mpg', y='price', data=df, ci=None)
plt.show()

# Residual plot
sns.residplot(x='highway_mpg', y='price', data=df)
plt.axhline(0, color='k', linestyle='--')
plt.show()

# Distribution plot (predicted vs actual)
y_pred = lm.predict(X)
sns.kdeplot(y, color='red', label='Actual', fill=True)
sns.kdeplot(y_pred, color='blue', label='Predicted', fill=True)
plt.legend()
plt.show()
```

---

## 4. Residuals: Patterns & What They Mean

- Random scatter around zero: linear model is appropriate.
- Systematic curvature: indicates non-linear relationship; consider polynomial regression or other non-linear models.
- Non-constant variance (heteroscedasticity): variance of residuals changes with x → consider transforms (log), weighted regression, or heteroscedasticity-robust inference.
- Large outliers: investigate data quality or influential points; consider robust regression or transform/remove problematic points.

---

## 5. Polynomial Regression

- Polynomial regression models nonlinear relationships by including powers of predictors while still using linear regression on transformed features.
- Example: Quadratic (2nd degree) SLR:
  ŷ = b0 + b1*x + b2*x^2
- Degree controls flexibility: higher degree → more flexible → risk of overfitting.

Python:
- For 1D polynomial fits, numpy.polyfit is convenient for quick exploratory fits.
- For multi-dimensional polynomial expansion, use scikit-learn's `PolynomialFeatures`.

Examples:
```python
# numpy polyfit (1D)
coeffs = np.polyfit(df['highway_mpg'], df['price'], deg=3)  # cubic
p = np.poly1d(coeffs)
# p(x) evaluates the polynomial

# scikit-learn polynomial features + linear regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)  # transforms [x] -> [x, x^2]

lm_poly = LinearRegression()
lm_poly.fit(X_poly, y)
```

Notes:
- Keep degree as small as needed to capture curvature.
- Cross-validate degree to avoid overfitting.

---

## 6. Preprocessing & Pipelines

- Preprocessing tasks often include scaling (StandardScaler), encoding categorical variables, and polynomial feature generation.
- Pipelines chain preprocessing and modeling steps to simplify code and avoid data leakage (fit/transform only on training data).
- Example pipeline: Standardization → PolynomialFeatures → LinearRegression.

Example:
```python
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linreg', LinearRegression())
])

pipe.fit(X, y)
y_pred = pipe.predict(X_new)
```

Benefits:
- Cleaner code
- Safe fit/transform sequence for cross-validation
- Easier hyperparameter tuning with GridSearchCV

---

## 7. Numerical Evaluation Metrics

1. Mean Squared Error (MSE)
   - MSE = mean((y - ŷ)^2)
   - Intuitive: average squared prediction error; penalizes large errors more heavily.
   - Implementation: `sklearn.metrics.mean_squared_error(y_true, y_pred)`

2. R-squared (R², coefficient of determination)
   - R² = 1 - (MSE_model / MSE_baseline)
   - Baseline is typically predicting the mean of y for all samples.
   - Interprets proportion of variance explained by the model (0 to 1 in typical well-behaved cases).
   - Use `model.score(X, y)` for LinearRegression to get R².

Interpretation:
- High R² and low MSE generally indicate good fit on the dataset, but:
  - Adding more variables always tends to decrease MSE and increase R² on training data — may cause overfitting.
  - Compare models using cross-validated metrics for a fair assessment.
  - R² can be negative if the model performs worse than predicting the mean (possible on test set or when overfitting/poor model).

Example:
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y, y_pred)
r2 = lm.score(X, y)  # or sklearn.metrics.r2_score(y, y_pred)
print("MSE:", mse)
print("R^2:", r2)
```

Guidance:
- Use cross-validation to get reliable estimates of generalization performance.
- Consider adjusted R² or information criteria (AIC/BIC) when comparing models with different numbers of features.
- For many problems, domain expectations matter: acceptable R² depends on the field (e.g., social sciences vs physical sciences).

---

## 8. Comparing SLR and MLR

- MLR includes more predictors; therefore on training data:
  - MSE typically decreases.
  - R² typically increases.
- However, a lower training MSE does not guarantee better generalization.
- Evaluate using cross-validation and holdout (test) data.
- Beware of multicollinearity in MLR — highly correlated predictors can make coefficient estimates unstable and hard to interpret.

---

## 9. Practical Tips & Diagnostics

- Always visualize data before modeling (scatter plots, pair plots, histograms).
- Plot residuals to check linearity assumptions and heteroscedasticity.
- Detect and handle outliers/influential points (Cook's distance, leverage).
- Standardize features when combining features with different scales (especially before regularization).
- Use pipelines to ensure reproducible preprocessing and avoid leakage.
- Regularize (Ridge, Lasso) when dealing with many correlated predictors to reduce overfitting.
- Use cross-validation and holdout test set for model assessment.
- Interpret coefficients carefully: in MLR, each coefficient represents the effect of that variable holding others constant.

---

## 10. Quick Checklist for Building a Regression Model

1. Understand the problem and target variable.
2. Visualize features vs target and distributions.
3. Split into train/test (and possibly validation).
4. Choose baseline model (mean predictor) and simple models first (SLR).
5. Fit model(s) and inspect coefficients.
6. Evaluate with MSE, R², and visual diagnostics (residuals, dist plots).
7. If residuals show non-linearity, try polynomial features or other non-linear models.
8. If performance is poor or overfitting occurs, consider:
   - More data
   - Regularization (Ridge/Lasso)
   - Feature selection or dimensionality reduction
9. Use pipelines + cross-validation for robust evaluation.
10. Report test-set metrics and diagnostics.

---

## 11. Example End-to-End Workflow (Code)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load or create dataframe df with features and 'price' target
# df = pd.read_csv('cars.csv')

X = df[['highway_mpg', 'curb_weight', 'age', 'engine_size']]  # example
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('reg', Ridge(alpha=1.0))  # Ridge as example of regularized linear model
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R^2:", r2_score(y_test, y_pred))

# Residual plot (using first feature for x-axis)
residuals = y_test - y_pred
sns.scatterplot(x=X_test['highway_mpg'], y=residuals)
plt.axhline(0, linestyle='--', color='k')
plt.xlabel('highway_mpg')
plt.ylabel('residual (y - ŷ)')
plt.show()
```

---

## 12. Summary

- Linear regression (simple or multiple) is an interpretable baseline for continuous prediction.
- Visual checks (regression plot, residual plot, distribution plots) are crucial to understand model fit and assumptions.
- Polynomial regression can model curvature; pipelines simplify chaining transforms and estimators.
- Use MSE and R² to quantify fit, but always check generalization using cross-validation and test sets.
- Watch for extrapolation, non-linearity, heteroscedasticity, outliers, and multicollinearity.