# Exploratory Data Analysis (EDA) — Quick Summary 📊🧠

A concise, friendly EDA reference that shows what to do, why it matters, and example code snippets for common tasks. Use this as a checklist while exploring new datasets.

---

## Table of contents
- ✅ What is EDA?
- 🔎 Goals of EDA
- 📋 Descriptive statistics
  - describe()
  - value_counts()
  - Box plot
  - Scatter plot
- 📚 GroupBy & pivot tables
- 🧪 ANOVA (Analysis of Variance)
- 🔗 Correlation & correlation statistics
- 🧰 Quick tips & best practices
- 🚀 Next steps

---

## ✅ What is EDA?
Exploratory Data Analysis (EDA) is the preliminary step to:
- Summarize main characteristics of the data
- Gain understanding of distributions and missingness
- Uncover relationships between variables
- Find candidate predictors and signals for modeling

Think of EDA as *investigation and storytelling* about your dataset. 🕵️‍♀️

---

## 🔎 Goals of EDA
- Understand variable types and distributions
- Detect outliers and anomalies
- Identify correlations and potential causation hypotheses (but remember: correlation ≠ causation) ⚠️
- Reduce dimensionality / highlight important variables

---

## 📋 Descriptive statistics

### describe() — numeric summary
Use pandas describe() to summarize numeric columns quickly.

```python
import pandas as pd
data = pd.read_csv("your_dataset.csv")
print(data.describe())
```

What describe() gives you:
- count, mean, std, min, 25%, 50% (median), 75%, max

Tip: use `data.describe(include="all")` to include non-numeric columns (counts, unique, top, freq). ✨

---

### value_counts() — categorical summary
Summarize categories and their frequency.

```python
category_counts = data["Category"].value_counts()
print(category_counts)
```

Add normalize=True to get proportions:
```python
data["Category"].value_counts(normalize=True)
```

---

### Box plot — distribution & outliers 📦
Boxplots show min, Q1, median, Q3, max and highlight outliers.

```python
import matplotlib.pyplot as plt
data.boxplot(column="NumericalColumn")
plt.title("Boxplot — NumericalColumn")
plt.ylabel("Value")
plt.show()
```

Use seaborn for prettier plots:
```python
import seaborn as sns
sns.boxplot(x="Category", y="NumericalColumn", data=data)
plt.show()
```

---

### Scatter plot — relationship between two numeric features 🔗
Visualize pairwise relationships and potential linear trends.

```python
plt.scatter(data["NumericalColumn1"], data["NumericalColumn2"], alpha=0.6)
plt.xlabel("NumericalColumn1")
plt.ylabel("NumericalColumn2")
plt.title("Scatter plot")
plt.show()
```

With seaborn and regression line:
```python
sns.regplot(x="NumericalColumn1", y="NumericalColumn2", data=data)
plt.ylim(0,)
plt.show()
```

---

## 📚 GroupBy & Pivot Tables

GroupBy: aggregate statistics by categorical variables.

Example — average price by drive_wheels and body_style:

```python
df_test = df[["drive_wheels", "body_style", "price"]]
df_grp = df_test.groupby(["drive_wheels", "body_style"], as_index=False).mean()
print(df_grp)
```

Pivot table: reshape the grouped table for easier comparison and plotting:

```python
df_pivot = df_grp.pivot(index="drive_wheels", columns="body_style", values="price")
print(df_pivot)
```

Heatmap visualization:
```python
plt.figure(figsize=(8,6))
sns.heatmap(df_pivot, annot=True, fmt=".0f", cmap="RdBu_r")
plt.title("Average Price by drive_wheels & body_style")
plt.show()
```

---

## 🧪 ANOVA (Analysis of Variance)

Purpose: compare the means of 3+ groups (or between two groups too) to see if differences are statistically significant.

Example — compare price distributions across makes (e.g., Honda vs Subaru):

```python
from scipy import stats

df_anova = df[["make", "price"]]
grouped = df_anova.groupby("make")

stat, pvalue = stats.f_oneway(grouped.get_group("honda")["price"],
                              grouped.get_group("subaru")["price"])
print("F-statistic:", stat, "p-value:", pvalue)
```

Interpretation:
- Low p-value (commonly < 0.05) → reject null hypothesis (means differ)
- ANOVA assumes normality and similar variances — check assumptions or use non-parametric alternatives.

---

## 🔗 Correlation

Correlation measures linear association between two numeric variables.

- Positive correlation: both variables increase together (e.g., engine-size ↑ → price ↑) 📈
- Negative correlation: one increases while the other decreases (e.g., highway-mpg ↑ → price ↓) 📉
- No correlation: no linear relationship

Plot examples:

Positive linear relationship:
```python
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
plt.show()
```

Negative linear relationship:
```python
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
plt.show()
```

Weak/no correlation:
```python
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
plt.show()
```

---

## Correlation — statistics

Pearson correlation coefficient r:
- r ∈ [-1, 1]
  - r = 1: perfect positive linear correlation
  - r = -1: perfect negative linear correlation
  - r = 0: no linear correlation

Also check p-value for significance.

Compute with scipy:

```python
from scipy.stats import pearsonr
r, p = pearsonr(df["engine-size"], df["price"])
print("r:", r, "p-value:", p)
```

Guidance:
- |r| > 0.7 often considered strong (context dependent)
- p-value < 0.05 commonly used as significant
- Strong correlation + small p-value suggests a reliable linear association — not proof of causation ⚠️

---

## 🧰 Quick tips & best practices
- Always check for missing values early: `data.isna().sum()` ✅
- Plot distributions: histograms, KDEs, and boxplots to understand skewness
- Transform skewed features (log, box-cox) before modeling if needed 📐
- Use pairplots or correlation heatmaps to explore many variables at once:
  ```python
  sns.pairplot(df[["price","engine-size","highway-mpg","curb-weight"]])
  ```
- Check assumptions of statistical tests (normality, homoscedasticity)
- When sample sizes differ widely between groups, be cautious with p-values
- Document discoveries and questions — EDA is iterative and narrative-driven 📝

---

## 🚀 Next steps
- Create a reproducible EDA notebook (Jupyter / Colab) with the plots and tests shown above
- Summarize findings into a short report: key correlations, suspicious outliers, and recommended feature transformations
- Use findings to design feature engineering and modeling strategy

---

If you'd like, I can:
- Turn this into a ready-to-run Jupyter Notebook (with the sample dataset structure) 🧾
- Create a printable one-page cheat sheet PDF ✂️

Which would you prefer next? 😊