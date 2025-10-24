# Model Development

A model can be thought of as a mathematical equation used to predict a value given one or more other values
Relating one or more independent variables (features) to a dependent variable (target)
### Example

Suppose you use a car’s highway miles per gallon (MPG) as an independent variable (feature) to predict its price (dependent variable / target). Generally, using more relevant features yields more accurate predictions. Adding features such as age, mileage, brand, and color can improve model performance.

Consider this situation: two nearly identical cars differ only by color — one is pink and one is red. If color affects price (pink cars sell for less) but color is not included as a feature, the model will predict the same price for both and produce an inaccurate estimate.

Besides collecting more relevant data, you can try different model types. In this course you will learn:
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression

### Simple vs. Multiple Linear Regression

#### Simple Linear Regression (SLR)
- Goal: model the relationship between one predictor `x` and one target `y`.
- Model form: y = b0 + b1 * x + ε  
    - `b0` = intercept, `b1` = slope (coefficient), `ε` = noise (random error).
- Training: given paired samples stored in `X` (shape `(n_samples, 1)`) and `y` (shape `(n_samples,)`), we fit the model to estimate `b0` and `b1`. The fitted model produces predictions denoted `ŷ`.
- Example: if the learned equation is Price = 38423.31 − 821.73 × highway_mpg, then for `highway_mpg = 20` the predicted price is 38423.31 − 821.73×20.
- Typical scikit-learn workflow:
    ```
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X, y)             # X shape (n_samples, 1)
    y_pred = lm.predict(X_new)
    intercept = lm.intercept_
    coef = lm.coef_[0]
    ```

#### Multiple Linear Regression (MLR)
- Goal: model one continuous target `y` using two or more predictors `x1, x2, ..., xp`.
- Model form: y = b0 + b1*x1 + b2*x2 + ... + bp*xp + ε  
    - `b0` = intercept, `b1..bp` = coefficients for each feature.
- Data representation: features stored in a matrix `Z` with shape `(n_samples, p)`; target in `y`.
- Training and prediction are analogous to SLR; coefficients explain expected change in `y` per one-unit change in each feature, holding others constant.
- scikit-learn example:
    ```
    lm = LinearRegression()
    lm.fit(Z, y)            # Z shape (n_samples, p)
    y_pred = lm.predict(Z_new)
    intercept = lm.intercept_
    coefs = lm.coef_        # array of length p
    ```

#### Practical notes
- The model is an estimate; prediction errors arise from noise, model misspecification, or omitted relevant features.
- Workflow summary: collect training points → fit model → obtain parameters → predict with `ŷ` → evaluate by comparing `ŷ` to actual `y`.
- When adding predictors, expect improved fit only if new features carry relevant information; otherwise you risk overfitting.
