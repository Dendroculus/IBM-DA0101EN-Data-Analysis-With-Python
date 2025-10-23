<h1 align="center">📊 IBM DA0101EN - Module 1 Quiz: Answers & Explanations 🐍</h1>

<p align="center">
  <em>This document provides the correct answers and explanations for the Module 1 quiz of the IBM Data Analysis with Python course. Understanding these basics is key to building a strong foundation!</em>
</p>

---

## ❓ Question 1
What does CSV stand for?

-   ✅ **Comma-separated values**
-   Car sold values
-   Car state values
-   None of the above

> **💡 Explanation:** CSV stands for **Comma-Separated Values**. It's a very common file format used in data analysis where data fields in a table are separated (delimited) by a comma.

---

## ❓ Question 2
In the data set, which of the following represents an attribute or feature?

-   Row
-   ✅ **Column**
-   Each element in the dataset

> **💡 Explanation:** In tabular data (like a spreadsheet or a Pandas DataFrame), each **Column** represents a specific **attribute** or **feature** being measured or described (e.g., 'age', 'price', 'city'). Each *Row* typically represents a single observation or record. 

---

## ❓ Question 3
What is the name of what we want to predict?

-   ✅ **Target**
-   Feature
-   Dataframe

> **💡 Explanation:** In predictive modeling and machine learning, the **Target** (or target variable, sometimes called the label or dependent variable) is the specific outcome or value we are trying to predict using the other features (predictors) in our dataset.

---

## ❓ Question 4
What is the command to display the first five rows of a dataframe `df`?

-   ✅ **df.head()**
-   df.tail()

> **💡 Explanation:** The `.head()` method in Pandas is specifically designed to show the first `n` rows of a DataFrame. By default, if you don't specify a number inside the parentheses, it shows the **first 5 rows**, which is great for quickly previewing your data. `.tail()` shows the *last* rows.

---

## ❓ Question 5
What command do you use to get the data type of each row of the dataframe `df`? (Note: The question likely meant *column*, as data types apply to columns/features).

-   ✅ **df.dtypes**
-   df.head()
-   df.tail()

> **💡 Explanation:** The `.dtypes` **attribute** (notice no parentheses `()`) in Pandas returns a Series containing the data type of each **column** in the DataFrame. Understanding the data types (like `int64`, `float64`, `object`, `bool`) is crucial before performing analysis.

---

## ❓ Question 6
How do you get a statistical summary of a dataframe `df`?

-   ✅ **df.describe()**
-   df.head()
-   df.tail()

> **💡 Explanation:** The `.describe()` method in Pandas is a powerful tool that generates descriptive statistics for the **numerical columns** in your DataFrame by default. This includes count, mean, standard deviation, minimum, maximum, and quartile values, giving you a quick overview of the data's distribution.

---

## ❓ Question 7
If you use the method `describe()` without changing any of the arguments, you will get a statistical summary of all the columns of type "object".

-   ✅ **False**
-   True

> **💡 Explanation:** By default, `df.describe()` **only provides statistics for numerical columns** (like `int` and `float`). To get a summary for columns of type "object" (which usually contain strings), you need to explicitly tell it to include them, like this: `df.describe(include='object')`. The summary for object columns includes count, unique values, the most frequent value (top), and its frequency (freq).

---

<p align="center">
  <em>Hope this helps!</em> 🎉
</p>