# Model Evaluation, Overfitting, Underfitting, Model Selection, Ridge Regression, and Grid Search — Comprehensive Notes

About these notes
- These notes are a distilled, structured summary of the provided transcript. They cover splitting data, generalization error, cross‑validation, model selection (polynomial degree), overfitting vs underfitting, ridge regression and regularization, hyperparameter selection via grid search, and practical code examples using scikit‑learn, numpy, pandas, and seaborn.
- Use the code snippets as 1) direct references for experimentation and 2) templates to integrate into Jupyter notebooks or pipelines.



## 1. Purpose of Model Evaluation
- Model evaluation estimates how well a trained model will perform on new, unseen data (the "real world").
- In‑sample (training) evaluation shows how well the model fits the data it was trained on; it does NOT guarantee real-world performance.
- Out‑of‑sample evaluation uses held-out data (test set or cross‑validation) to estimate generalization error.



## 2. Train / Test Split
- Split dataset into a training set (used to fit the model) and a testing set (used to estimate out‑of‑sample performance).
- Typical splits: 70/30, 80/20, 90/10 (training/test). Larger training set gives better accuracy estimate but higher variance in performance estimate; larger test set improves precision of the estimate but reduces training data.
- Use a fixed random seed for reproducible splitting.

Example:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
```

Practical note:
- After you finalize hyperparameters, retrain on the entire dataset (train+test) before deployment if appropriate.



## 3. Generalization Error, Bias & Variance Intuition
- Generalization error: expected prediction error on new data.
- Tradeoff:
  - High bias (underfitting): model too simple, large error on both training and test sets.
  - High variance (overfitting): model too complex, low training error but large test error.
- Goal: find model complexity / hyperparameters that minimize test error (balance bias and variance).

Bullseye analogy:
- Center = true generalization error.
- Many trials with different train/test splits cluster around true value (good accuracy); variability between trials is precision.



## 4. Cross‑Validation (CV)
- CV reduces variability of single train/test split estimates by averaging across multiple splits.
- K‑fold CV: split data into k folds; train on k‑1 folds and test on the remaining fold; repeat so each fold is used once for testing; average metric over folds.
- Typical values: k=5 or k=10. For small datasets, consider leave‑one‑out (LOO) or larger k.

scikit‑learn examples:
```python
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()

# cross_val_score returns scores (default score is R^2 for regressors)
scores = cross_val_score(model, X, y, cv=5)  # returns array of 5 scores
mean_score = np.mean(scores)

# cross_val_predict returns the out‑of‑fold predictions for each sample
y_oof = cross_val_predict(model, X, y, cv=5)
```

Notes:
- Use cross_val_predict when you need the predicted values (e.g., to plot residuals or predicted vs actual).
- For classification with imbalanced classes use StratifiedKFold; for regression, consider KFold.



## 5. Underfitting vs Overfitting (visual & numeric signs)
- Underfitting:
  - Signs: model performs poorly on training and test sets.
  - Residuals: large systematic errors; structure not captured.
- Overfitting:
  - Signs: great performance on training set, poor on test set.
  - Residuals: small on train, large and structured on test.
- Use learning curves and validation curves to diagnose.

Validation curve behavior (example with polynomial degree):
- Train error decreases monotonically as model complexity increases.
- Test error decreases until an optimal complexity and then increases (overfitting region).



## 6. Model Selection: Polynomial Degree Example
- Task: choose polynomial order that best approximates underlying function (balance fit and generalization).
- Procedure:
  1. For each degree d in candidate list:
     - Transform features with PolynomialFeatures(degree=d).
     - Fit model on training set.
     - Evaluate on validation/test set (or use CV).
  2. Plot test metric (MSE or R²) vs degree and pick degree that minimizes test error (or maximizes R²).
- Beware: very high degree can oscillate and chase noise (Runge phenomenon).

Example: evaluating R² for different degrees
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

degrees = range(1, 11)
r2_test = []

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_d = poly.fit_transform(X_train[['horsepower']])
    X_test_d = poly.transform(X_test[['horsepower']])
    lm = LinearRegression().fit(X_train_d, y_train)
    r2_test.append(r2_score(y_test, lm.predict(X_test_d)))
```

Plot r2_test vs degrees to find optimal degree (peak R²).



## 7. Residuals, Distribution Plots & Diagnostics
- Residual plot:
  - x-axis: predictor (or predicted value)
  - y-axis: residuals (y - ŷ)
  - Good linear model: residuals randomly scattered around 0 with constant variance (homoscedastic).
  - Bad signs: curvature, funnel shape, clusters → indicates nonlinearity, heteroscedasticity, or missing variables.
- Distribution (density/kde) plot:
  - Compare distribution of predicted values vs actual. Useful for multidimensional models where single scatter plots are not feasible.
- Other helpful diagnostics:
  - Predicted vs actual scatter (identity line y=ŷ).
  - Cook's distance / leverage to detect influential points.
  - Learning curves and validation curves.

Example plotting:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# residuals
y_pred = model.predict(X_test)
residuals = y_test - y_pred
sns.scatterplot(x=X_test['horsepower'], y=residuals)
plt.axhline(0, color='k', linestyle='--')

# distribution
sns.kdeplot(y_test, color='red', label='actual', fill=True)
sns.kdeplot(y_pred, color='blue', label='predicted', fill=True)
plt.legend()
```



## 8. Ridge Regression (L2 Regularization)
- Purpose: reduce overfitting by penalizing large coefficient magnitudes.
- Ridge objective: minimize (RSS + α * ||w||^2).
  - α (alpha) controls strength of penalty:
    - α = 0 → ordinary least squares (no regularization).
    - α → large → coefficients shrink toward zero (increased bias, decreased variance).
- Effect:
  - Stabilizes coefficients, especially when features are highly correlated or when using high‑degree polynomial features.
  - Reduces variance at the cost of increased bias; tuned α balances the tradeoff.

Fitting Ridge:
```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
```

Coefficient shrinkage demonstration:
- As α increases, higher‑order polynomial coefficients' magnitudes decrease.
- Excessive α leads to underfitting.

Recommendation:
- Standardize features before Ridge (use StandardScaler or pipeline).



## 9. Hyperparameter Selection & Validation Data
- Hyperparameters (e.g., polynomial degree, ridge alpha) are not learned during fitting; they must be chosen via validation or cross‑validation.
- Common approach:
  - Split data: training, validation (sometimes via CV), and test.
  - Use training set to fit models for candidate hyperparameters.
  - Use validation set (or CV) to select best hyperparameter.
  - Report final performance on test set.
- Alternatively, use K‑fold CV across hyperparameter grid (GridSearchCV) and evaluate best model on a held‑out test set afterwards.



## 10. Grid Search with Cross‑Validation (GridSearchCV)
- Grid search automates hyperparameter tuning by exhaustively searching specified parameter values combined with cross‑validation.
- Input: estimator, param_grid (dictionary or list of dicts), cv (number of folds), scoring metric (e.g., 'r2', 'neg_mean_squared_error').
- After fit, useful attributes:
  - best_estimator_, best_params_, best_score_
  - cv_results_ (detailed information for each candidate)

Example:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('ridge', Ridge())
])

param_grid = {
    'poly__degree': [1, 2, 3, 4],
    'ridge__alpha': [0.001, 0.01, 0.1, 1, 10]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)      # mean cross-validation score for best params
cv_results = grid.cv_results_ # dict with detailed per-parameter scores
```

Notes:
- Prefer scaling via StandardScaler in a pipeline rather than estimator's `normalize` parameter (which is deprecated in newer scikit‑learn versions).
- GridSearchCV performs nested loops: for each parameter combination, it performs CV on training data to produce validation score.



## 11. Practical Example: Selecting Alpha for Ridge Using Validation Curve
- Iterate α values, record train and validation R² (or MSE), and plot them vs α.
- Select α that maximizes validation R² (or minimizes validation MSE).

Example:
```python
import numpy as np
from sklearn.model_selection import cross_val_score

alphas = np.logspace(-4, 4, 10)
train_scores = []
val_scores = []

for a in alphas:
    ridge = Ridge(alpha=a)
    # CV on training set to estimate validation performance
    scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
    val_scores.append(np.mean(scores))
    # Optionally compute training score directly for diagnostic
    ridge.fit(X_train, y_train)
    train_scores.append(ridge.score(X_train, y_train))

# plot train_scores and val_scores vs alphas (log scale)
```

Interpretation:
- If train_score ≈ val_score and both low → underfitting.
- If train_score high and val_score low → overfitting.



## 12. Nested Cross‑Validation (brief)
- When tuning hyperparameters and estimating generalization performance, nested CV avoids optimistic bias:
  - Outer CV loop estimates generalization error.
  - Inner CV loop performs hyperparameter selection (GridSearchCV).
- Use nested CV when you need an unbiased estimate of performance for model selection.



## 13. Practical Workflow Checklist
1. Explore and visualize data (scatter, histograms, pairplots).
2. Clean and preprocess (handle missing values, encode categoricals, scale features).
3. Split data (train/test) or plan CV strategy.
4. Start with simple baseline models (e.g., mean predictor, SLR).
5. Fit models and inspect coefficients (interpretability).
6. Use diagnostics: residual plots, predicted vs actual, distribution plots.
7. If nonlinearity suspected, try polynomial features or more flexible models.
8. Use cross‑validation to estimate out‑of‑sample performance.
9. Tune hyperparameters (alpha, degree, regularization strength) using GridSearchCV or RandomizedSearchCV with CV.
10. Inspect cv_results_ to understand sensitivity to hyperparameters.
11. Check for overfitting or high variance; add regularization, gather more data, or reduce model complexity.
12. Once final hyperparameters chosen, evaluate on held‑out test set (or perform nested CV for unbiased estimate).
13. Retrain final model on full dataset if deploying (optionally after careful validation).



## 14. Common Pitfalls & Tips
- Do not evaluate model performance on data used for training/hyperparameter tuning — use separate test or nested CV.
- Avoid selecting hyperparameters based on test set performance; use validation/CV instead.
- Always scale features before regularized regression (Ridge, Lasso).
- Prefer pipelines to chain preprocessing and modeling; pipelines prevent data leakage during CV.
- Be cautious when extrapolating outside the training data range (predictions may be unrealistic).
- Monitor interpretability: in MLR, coefficients represent marginal effects holding other features constant — multicollinearity can make them unstable.
- Use learning curves (varying training size) to diagnose whether collecting more data would help.



## 15. Useful scikit‑learn Tools & Functions
- train_test_split — split data into train and test sets.
- cross_val_score — compute cross‑validated metric scores.
- cross_val_predict — obtain out‑of‑fold predictions.
- GridSearchCV / RandomizedSearchCV — hyperparameter search with CV.
- PolynomialFeatures — expand features to polynomial terms.
- Pipeline — chain preprocessing and estimator steps safely.
- Ridge — L2 regularized linear regression.
- sklearn.metrics: mean_squared_error, r2_score — evaluation metrics.



## 16. Example End‑to‑End (compact)
```python
# Example pipeline with polynomial features, scaling, Ridge and GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('ridge', Ridge())
])

param_grid = {
    'poly__degree': [1, 2, 3],
    'ridge__alpha': [0.001, 0.01, 0.1, 1, 10]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_test_pred = best_model.predict(X_test)

print("Test R^2:", r2_score(y_test, y_test_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Best params:", grid.best_params_)
```



## 17. Summary
- Use train/test split and cross‑validation to estimate generalization performance.
- Diagnose underfitting/overfitting via residuals, learning/validation curves, and comparing train vs validation metrics.
- Use polynomial features carefully; choose degree by minimizing test/CV error (not by training error).
- Regularize (Ridge) to control coefficient magnitudes and reduce overfitting — tune α via cross‑validation.
- Automate hyperparameter tuning using GridSearchCV (or RandomizedSearchCV) within a pipeline and evaluate final performance on held‑out data.
- Prefer pipelines and cross‑validation to avoid leakage and obtain reliable performance estimates.

 
