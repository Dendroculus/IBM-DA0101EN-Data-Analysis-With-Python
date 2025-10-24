# Model Development — Comprehensive Notes

A model (estimator) is a mathematical function used to predict a value (target) from one or more inputs (features). These notes cover regression modeling end-to-end: theory, practical workflows, diagnostics, feature engineering, model selection, uncertainty quantification, deployment, and a used‑car valuation workflow with runnable examples.

Table of contents
1. Goals & Definitions
2. Assumptions & When Linear Models Make Sense
3. Simple Linear Regression (SLR)
4. Multiple Linear Regression (MLR)
5. Polynomial Regression & Interaction Terms
6. Metrics & Model Comparison
7. Visual Diagnostics
8. Validation: Train/Test, K-Fold, Nested CV, and Bootstrap
9. Preprocessing, Pipelines & ColumnTransformer
10. Categorical Encoding & High‑Cardinality Features
11. Feature Engineering & Selection (including interactions)
12. Multicollinearity & VIF
13. Regularization: Ridge, Lasso, ElasticNet
14. Nonlinear & Tree‑Based Models, Model Explanation (SHAP/PDP)
15. Prediction Intervals & Uncertainty Quantification
16. Robustness: Outliers, Heteroscedasticity, Autocorrelation
17. Model Selection, Hyperparameter Tuning & Information Criteria
18. Deployment, Monitoring & Reproducibility
19. Practical Used‑Car Valuation Workflow
20. Checklist Before Making Decisions
21. Suggested Exercises & Example Code Snippets
22. References & Further Reading

---

1. Goals & Definitions
- Feature (independent variable, x): input used to predict the target.
- Target (dependent variable, y): variable we want to predict (e.g., car price).
- Model: f(x; θ) ≈ y. For linear regression: y ≈ b0 + b1 x1 + ... + bp xp.
- Prediction ŷ: model’s estimate for a given x.
- Residual: e = y − ŷ.

Use cases:
- Prediction (accuracy on unseen data).
- Inference (interpret coefficients and test hypotheses).
- Decision making (use predictions + uncertainty + business rules).

---

2. Assumptions & When Linear Models Make Sense
Classical linear regression assumptions (useful checklist for interpreting p-values and intervals):
- Linearity: E[y|x] is linear in parameters.
- Independence of errors: no dependence between residuals.
- Homoscedasticity: constant variance of residuals.
- Normality of residuals (for small samples and exact inference).
- No (or manageable) multicollinearity.
When assumptions are violated, consider transformations, robust methods, or non‑linear models.

---

3. Simple Linear Regression (SLR)
- Model: y = b0 + b1 x + ε.
- Fit with OLS (ordinary least squares) to minimize sum of squared residuals.
- Interpretation: b1 = expected change in y per unit increase in x.

sklearn example:
```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X.reshape(-1,1), y)
print("intercept", lm.intercept_, "slope", lm.coef_[0])
y_pred = lm.predict(X_new.reshape(-1,1))
```

statsmodels for inference (p-values, CI):
```python
import statsmodels.api as sm
X_const = sm.add_constant(X)
res = sm.OLS(y, X_const).fit()
print(res.summary())
```

---

4. Multiple Linear Regression (MLR)
- Model: y = b0 + b1 x1 + b2 x2 + ... + bp xp + ε.
- Coefficients interpreted ceteris paribus (holding other features constant).
- Beware omitted-variable bias: missing a relevant predictor that correlates with both included features and the target can bias coefficients.

sklearn:
```python
lm = LinearRegression().fit(X_train, y_train)
y_pred = lm.predict(X_test)
```

---

5. Polynomial Regression & Interaction Terms
- Polynomial regression: include polynomial terms (x^2, x^3...) to model curvature while staying linear in parameters.
- Interaction terms: x1 * x2 to model conditional effects.
- Use sklearn.preprocessing.PolynomialFeatures to generate interaction and polynomial terms.
- Important: polynomial features explode dimensionality — use regularization and CV.

Example:
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

---

6. Metrics & Model Comparison
Primary metrics for regression:
- MSE = mean((y − ŷ)^2)
- RMSE = sqrt(MSE) — same units as y
- MAE = mean(|y − ŷ|) — robust to outliers
- Median Absolute Error — robust
- R² = 1 − SS_res/SS_tot (proportion of variance explained)
- Adjusted R² = 1 − (1−R²)*(n−1)/(n−p−1) — penalizes adding irrelevant features
- MAPE (mean absolute percentage error) — beware near-zero targets

Use cross-validated metrics (CV-RMSE, CV-MAE) for model selection. Prefer MAE or RMSE depending on cost function (MAE is more robust; RMSE penalizes large errors).

---

7. Visual Diagnostics
- Scatter + fitted line (SLR): quick check for linearity & outliers.
- Residuals vs Fitted: diagnose nonlinearity & heteroscedasticity.
- Q‑Q plot of residuals: normality check.
- Scale-Location plot (sqrt(|residuals|) vs fitted): heteroscedasticity.
- Influence plots (leverage vs residuals) and Cook’s distance: detect influential points.
- Distribution plot of predictions vs actual: see calibration and bias.

Example (matplotlib + seaborn):
```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x=X.ravel(), y=y, alpha=0.6)
plt.plot(X_sorted, model.predict(X_sorted.reshape(-1,1)), color='red')
```

---

8. Validation: Train/Test, K-Fold, Nested CV, and Bootstrap
- Train/test split: basic sanity check (80/20 or 70/30).
- K‑Fold CV: K=5 or 10. Use for estimating generalization error reliably.
- Stratify when target is categorical; for regression consider stratifying by binned target.
- Nested CV: use when selecting hyperparameters to avoid optimistic estimates (outer loop for test error, inner loop for hyperparameter tuning).
- Bootstrap: estimate variability of predictions/parameters and build empirical prediction intervals.

sklearn cross-validation:
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-scores).mean()
```

---

9. Preprocessing, Pipelines & ColumnTransformer
- Always fit preprocessing only on training data to avoid leakage.
- Use Pipeline to chain transformations and estimator, and ensure transforms are applied consistently at predict time.
- Use ColumnTransformer to apply different preprocessing to numeric and categorical features.

Example:
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_pipe = Pipeline([('scaler', StandardScaler())])
preproc = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
pipe = Pipeline([('preproc', preproc), ('model', LinearRegression())])
pipe.fit(X_train, y_train)
```

---

10. Categorical Encoding & High‑Cardinality Features
- Low-cardinality: OneHotEncoder (sparse matrices if many categories).
- High-cardinality: target encoding (mean encoding) or frequency encoding. Always guard against leakage: compute encodings using out-of-fold or smoothing.
- Ordinal variables: map to integers preserving order.
- Embeddings: for very high-cardinality features and complex models (neural networks), learned embeddings are powerful.

Target encoding (safe approach):
- Use K‑fold within training set to compute target means per category and avoid leakage.

---

11. Feature Engineering & Selection (including interactions)
- Derived features often improve performance: age = current_year − year, mileage_per_year, log(mileage).
- Binning continuous variables can capture nonlinearity but might lose information.
- Interaction features where domain suggests multiplicative effects (e.g., mileage * condition).
- Automated selection: recursive feature elimination, model-based selection (Lasso), tree-based importances, permutation importance.

---

12. Multicollinearity & VIF
- Multicollinearity inflates coefficient variance and complicates interpretation.
- Compute Variance Inflation Factor (VIF) for each predictor: VIF_i = 1/(1 − R_i^2).
- Rule of thumb: VIF > 5–10 indicates concern.
- Remedies: drop variables, combine them, use PCA, or use regularization (Ridge).

Example (VIF):
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_with_const = sm.add_constant(X)
vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

---

13. Regularization: Ridge, Lasso, ElasticNet
- Ridge (L2): shrinks coefficients continuously; good for multicollinearity.
- Lasso (L1): performs variable selection by zeroing coefficients.
- ElasticNet: combination of L1 and L2.
- Use cross-validation to select regularization strength (alpha) and mixing ratio.

Example:
```python
from sklearn.linear_model import RidgeCV, LassoCV
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5).fit(X_train, y_train)
lasso = LassoCV(cv=5).fit(X_train, y_train)
```

---

14. Nonlinear & Tree‑Based Models, Model Explanation (SHAP/PDP)
- Tree-based: DecisionTree, RandomForest, GradientBoosting (XGBoost, LightGBM, CatBoost).
- Tree ensembles often beat linear models for complex patterns and handle mixed data types.
- Explain tree models with:
  - Feature importances (global),
  - Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE),
  - SHAP values: provide per-observation explanations and consistent local/global importance.

SHAP example:
```python
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample)
```

---

15. Prediction Intervals & Uncertainty Quantification
- Analytical intervals: statsmodels OLS provides confidence and prediction intervals for linear models.
- Bootstrap: fit many bootstrap samples and derive percentile intervals for predictions.
- Quantile regression: predict conditional quantiles (e.g., 5th and 95th percentiles) — sklearn supports QuantileRegressor; gradient boosting frameworks support quantile loss.
- Bayesian regression: produce full posterior predictive distribution (PyMC, Stan).

statsmodels prediction interval example:
```python
model_sm = sm.OLS(y_train, sm.add_constant(X_train)).fit()
pred = model_sm.get_prediction(sm.add_constant(X_new))
print(pred.summary_frame(alpha=0.05))  # mean, mean_ci_lower/upper, obs_ci_lower/upper
```

Bootstrap prediction interval: (conceptual)
```python
preds = []
for i in range(n_boot):
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    model.fit(X_train[idx], y_train[idx])
    preds.append(model.predict(X_new))
preds = np.array(preds)
lower = np.percentile(preds, 2.5, axis=0)
upper = np.percentile(preds, 97.5, axis=0)
```

---

16. Robustness: Outliers, Heteroscedasticity, Autocorrelation
- Outliers: detect (z-score, IQR, Cook’s distance). Decide case-by-case whether to remove, transform, or use robust methods (HuberRegressor, RANSAC).
- Heteroscedasticity: Breusch-Pagan test; fix via weighted least squares or transform target (log).
- Autocorrelation (time series): Durbin‑Watson test; if present, use time‑aware models or GLS.

---

17. Model Selection, Hyperparameter Tuning & Information Criteria
- Use CV-based metrics for selection; for nested tuning use nested CV.
- GridSearchCV and RandomizedSearchCV for hyperparameters; for expensive searches consider Bayesian optimization (Optuna, scikit-optimize).
- Information criteria (AIC, BIC) are useful with likelihood-based models (statsmodels) to compare nested models with penalty for complexity.

Example GridSearch:
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'lm__alpha': [0.01, 0.1, 1.0, 10.0]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
```

---

18. Deployment, Monitoring & Reproducibility
- Save model + preprocessing pipeline (joblib, pickle) and version artifacts.
- Track model training dataset version, feature engineering code, random seeds, and hyperparameters (MLflow, DVC).
- Monitoring: log prediction distributions, error metrics, and data drift indicators (Population Stability Index, KL divergence, KS test).
- Retrain policy: schedule-based, drift-triggered, or performance threshold-based.

Save pipeline:
```python
import joblib
joblib.dump(pipe, "car_price_pipeline.joblib")
```

---

19. Practical Used‑Car Valuation Workflow
1. Data: collect year, make, model, trim, mileage, condition, location, engine size, transmission, color, accident_history, service_history, options.
2. Feature engineering: age, mileage_per_year, depreciation buckets, recent_service_flag, regional price index.
3. Preprocessing: impute, scale numeric features; encode categorical — treat color if it impacts price.
4. Model baseline: SLR using mileage or age. Compare to MLR with many features.
5. Diagnostics: residuals, influence, heteroscedasticity by price segments.
6. Model choice: linear for interpretability, tree ensemble for accuracy.
7. Uncertainty: bootstrap prediction intervals or quantile regression for actionable buy/sell ranges.
8. Combine with comps: nearest-neighbor averaging of recent comparable sales weighted by similarity and recency.
9. Business rule: recommend buy if predicted price − offer ≥ target_margin and lower bound of 95% PI still yields margin.

Example decision rule pseudocode:
```text
pred, lower, upper = model_predict_with_interval(vehicle)
if pred - offer >= target_margin and lower - offer >= min_margin:
    action = "buy"
else:
    action = "skip"
```

---

20. Checklist Before Making Decisions
- Data quality checks: missingness, duplicates, impossible values.
- Feature completeness and sensible transformations.
- Train/test split and CV applied.
- Residual diagnostics done and documented.
- Multicollinearity examined and addressed.
- Model compared to simple baselines (mean, median, kNN).
- Prediction uncertainty estimated and displayed.
- Interpretability checks (coefficients, SHAP).
- Business rule integration completed.

---

21. Suggested Exercises & Example Code Snippets
Exercises:
- Build SLR: highway_mpg → price. Plot residuals, compute RMSE & R².
- Build MLR: include year, mileage, make, color. Use pipelines.
- Polynomial expansion for mileage and test degrees 1..5 using CV.
- Regularization: run RidgeCV and LassoCV; compare coefficients and CV-RMSE.
- Use bootstrap to compute 95% PI for sample vehicles.
- Train RandomForestRegressor and compare with linear model; explain with SHAP.

Key code snippets (scikit-learn pipeline + GridSearch):
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

num_cols = ['mileage', 'age']
cat_cols = ['make', 'model', 'color']

preproc = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
])

pipe = Pipeline([
    ('preproc', preproc),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Ridge())
])

param_grid = {'model__alpha': [0.01, 0.1, 1.0, 10.0]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
```

---

22. References & Further Reading
- scikit-learn documentation — Pipelines, ColumnTransformer, model selection.
- statsmodels — OLS summaries, prediction intervals, hypothesis tests.
- Hastie, Tibshirani, Friedman — "The Elements of Statistical Learning".
- Géron — "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow".
- SHAP documentation for explainability.

---

Quick cheatsheet (condensed)
- Start simple: baseline SLR → MLR → polynomial/interactions → tree ensembles.
- Always use pipelines and CV.
- Prefer RMSE/MAE on validation for model selection.
- Regularize when p large or multicollinearity present.
- Use bootstrap or quantile regression for prediction intervals.
- Explain complex models with SHAP; check residuals & influencers for linear models.
- Combine model outputs with comps and business rules for final pricing.

---

If you'd like, I can:
- produce a runnable Jupyter notebook implementing the suggested exercises (data preprocessing, SLR, MLR, polynomial, Ridge/Lasso, bootstrap PI, Random Forest + SHAP),
- or merge this into your existing markdown file (tell me filename or repo),
- or create a one-page printable cheatsheet.
