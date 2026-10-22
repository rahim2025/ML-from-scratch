# Pandas Revision Sheet (for ML)

Each topic follows the same pattern so you can revise fast:

- **Theory** — what it is, why it matters, key rules
- **Example** — runnable code + the real output it produces

Most examples reuse **one shared sample dataset** called `df` (created in section 2), so you can watch the same data flow through *inspection → cleaning → feature selection → encoding → scaling*, just like a real ML pipeline.

## Table of Contents

1. [What is Pandas?](#1-what-is-pandas)
2. [Core Data Structures](#2-core-data-structures)
3. [Basic Inspection](#3-basic-inspection)
4. [Selecting Columns](#4-selecting-columns)
5. [Selecting Rows (loc vs iloc vs at/iat)](#5-selecting-rows-loc-vs-iloc-vs-atiat)
6. [Filtering Data](#6-filtering-data-boolean-masks)
7. [Missing Values](#7-missing-values)
8. [Duplicates](#8-duplicates)
9. [Modifying Data](#9-modifying-data)
10. [Applying Functions](#10-applying-functions-apply--map--lambda)
11. [Grouping & Aggregation](#11-grouping--aggregation-groupby--split--apply--combine)
12. [Combining DataFrames](#12-combining-dataframes)
13. [Feature Selection (X, y)](#13-feature-selection-ml-important)
14. [Encoding Categorical Data](#14-encoding-categorical-data)
15. [Train-Test Split](#15-train-test-split-with-sklearn)
16. [Feature Scaling](#16-feature-scaling)
17. [ML Workflow Summary](#17-ml-workflow-summary)
18. [Quick Memory Tricks](#18-quick-memory-tricks)

```python
import numpy as np
import pandas as pd
```

---

## 1. What is Pandas?

**Theory**

- Pandas = Python library for working with tabular/structured data (think: Excel sheets or SQL tables, but in Python).
- Terminology:
  - **Row** = sample / observation / record
  - **Column** = feature / variable / attribute
- Typical uses in an ML pipeline:
  1. Loading data (csv, excel, sql, json, ...)
  2. Cleaning data (missing values, duplicates, wrong types)
  3. Exploring data (statistics, grouping, sorting)
  4. Engineering/selecting features
  5. Producing the final `X`, `y` arrays that feed into a model
- Why it matters: almost no ML workflow skips pandas — it's the bridge between "raw messy data" and "clean numeric arrays" that scikit-learn / numpy / torch actually consume.

**Example**

```python
print(pd.DataFrame({"Age": [23, 22], "CGPA": [3.4, 3.8]}))
```

Output:

```
   Age  CGPA
0   23   3.4
1   22   3.8
```

Each **row** = one student (a "sample"). Each **column** = one attribute (a "feature") about that student.

---

## 2. Core Data Structures

**Theory**

- **Series** = 1D labeled array (one column). Has an index (labels) and values — think of it as a single Excel column with row labels.
- **DataFrame** = 2D table (rows + columns). Internally it's basically a dict of Series that share the same index.
- Every Series/DataFrame has an **index** — the row labels. By default it's `0, 1, 2, ...` but it can be custom (dates, IDs, names).
- Label lookup (`s["b"]`) uses the **index**, not the position.

**Example**

```python
# --- Series (1D): a single labeled column / array of values ---
s = pd.Series([10, 20, 30])
print(s)
# -> prints index 0,1,2 next to values 10,20,30, plus the dtype

s = pd.Series([10, 20, 30], index=["a", "b", "c"])      # custom index
print(s["b"])                                           # -> 20  (label lookup)

# --- DataFrame (2D): a table of rows & columns (the main structure) ---
# This is THE shared sample dataset reused for the rest of this sheet.
# It's intentionally a bit messy: Farhan's CGPA is missing, and row 4
# is an exact duplicate of row 1 -- perfect for practising cleaning.
df = pd.DataFrame({
    "Name":   ["Aisha", "Karim", "Farhan", "Nabila", "Karim"],
    "Age":    [23, 22, 25, 24, 22],
    "CGPA":   [3.4, 3.8, np.nan, 3.9, 3.8],
    "Gender": ["Female", "Male", "Male", "Female", "Male"],
    "Salary": [50000, 60000, 55000, 62000, 60000],
})
print(df)
```

```
     Name  Age  CGPA  Gender  Salary
0   Aisha   23   3.4  Female   50000
1   Karim   22   3.8    Male   60000
2  Farhan   25   NaN    Male   55000
3  Nabila   24   3.9  Female   62000
4   Karim   22   3.8    Male   60000
```

Other common ways to build / load a DataFrame:

```python
print(pd.DataFrame([[23, 3.4], [22, 3.8]], columns=["Age", "CGPA"]))

# pd.read_csv("data.csv")          # from a CSV file   (shown for syntax -- no file here)
# pd.read_excel("data.xlsx")       # from an Excel file
# pd.read_json("data.json")        # from a JSON file

# Saving a DataFrame back out
# df.to_csv("out.csv", index=False)
# df.to_excel("out.xlsx", index=False)
```

---

## 3. Basic Inspection

**Theory**

- These are usually the **first commands** you run on any new dataset: get a feel for size, structure, types, and a quick statistical summary before doing anything else.
- `head()` / `tail()` — peek at the data without printing everything
- `shape` — `(n_rows, n_columns)`
- `columns` — list of column names
- `info()` — dtypes + non-null counts + memory usage (great for spotting missing values & wrong types)
- `describe()` — count/mean/std/min/max/quartiles for numeric columns
- `unique()` / `nunique()` / `value_counts()` — understand categorical/discrete columns

**Example**

```python
print(df.head(3))      # first 3 rows           -> Aisha, Karim, Farhan
print(df.tail(2))      # last 2 rows            -> Nabila, Karim
print(df.shape)        # (n_rows, n_columns)    -> (5, 5)
print(df.columns)      # column names           -> Name, Age, CGPA, Gender, Salary
df.info()              # dtypes + non-null counts + memory (prints directly)
print(df.describe())   # stats for numeric cols -> count, mean, std, min, max...

print(df["Age"].unique())         # -> [23 22 25 24]            (4 distinct ages)
print(df["Age"].nunique())        # -> 4
print(df["Age"].value_counts())   # -> 22 appears twice, the rest appear once
```

---

## 4. Selecting Columns

**Theory**

- `df["col"]` → returns a **Series** (1D, single column)
- `df[["col"]]` → returns a **DataFrame** (2D, still a table!)
- `df[["c1", "c2"]]` → returns a DataFrame with multiple columns
- `df.col` → attribute-style shortcut, only works if the column name has no spaces/special characters
- Rule of thumb: **single brackets = Series, double brackets = DataFrame.**

**Example**

```python
print(df["Age"])                   # single column  -> Series of 5 ages
print(df[["Age"]])                 # single column  -> DataFrame (note double brackets)
print(df[["Age", "CGPA"]])         # multiple columns -> 2-column DataFrame
print(df.Age)                      # attribute-style access (only if name has no spaces)
```

---

## 5. Selecting Rows (loc vs iloc vs at/iat)

**Theory**

- `iloc` → integer **POSITION** based, like Python list slicing (end of a slice is **excluded**).
- `loc` → **LABEL** based, uses index/column names (end of a slice is **included**, because it refers to a label, not a position).
- `at` / `iat` → fast access to a **single scalar value** (label / position respectively). Faster than `loc`/`iloc` for one cell.
- ⚠️ **Gotcha:** `iloc[0:2]` returns 2 rows (end excluded) but `loc[0:2]` returns 3 rows (end included) — trips up almost everyone!

**Example**

```python
# --- iloc: integer POSITION based (end EXCLUDED, like list slicing) ---
print(df.iloc[0])             # first row              -> Aisha
print(df.iloc[0:2])           # rows at positions 0,1  -> Aisha, Karim   (2 rows)
print(df.iloc[0, 1])          # row-pos 0, col-pos 1   -> 23 (Aisha's Age)
print(df.iloc[:, 0])          # all rows, first column -> the Name column

# --- loc: LABEL based (uses index/column names, end INCLUDED) ---
print(df.loc[0])                     # row with index label 0 -> Aisha
print(df.loc[0:2])                   # labels 0,1,2 -> Aisha, Karim, Farhan (3 rows!)
print(df.loc[0, "Age"])              # row label 0, column "Age" -> 23
print(df.loc[:, ["Age", "CGPA"]])    # all rows, selected columns by name

# --- at / iat: fast access to a SINGLE scalar value ---
print(df.at[0, "Age"])        # label based, single value    -> 23
print(df.iat[0, 0])           # position based, single value -> "Aisha"
```

---

## 6. Filtering Data (boolean masks)

**Theory**

- A **boolean mask** is a Series of True/False, one per row, built by comparing a column to a value (`df["Age"] > 22`).
- `df[mask]` keeps only the rows where the mask is True.
- Combine conditions with `&` (and), `|` (or), `~` (not) — **not** the plain Python `and`/`or`/`not`. Always wrap each condition in parentheses because `&`/`|` bind tighter than comparison operators.
- `isin([...])` → row matches ANY value in a list
- `str.contains(...)` → substring match on a text column
- `df.query("...")` → write the same filter as a readable string

**Example**

```python
print(df[df["Age"] > 22])                       # -> Aisha, Farhan, Nabila (ages 23,25,24)
print(df[df["CGPA"] > 3.5])                     # -> Karim, Nabila, Karim  (CGPA 3.8,3.9,3.8)

# Combine conditions: use & (and), | (or), ~ (not) -- always with parentheses!
print(df[(df["Age"] > 22) & (df["CGPA"] > 3.5)])  # AND -> only Nabila satisfies BOTH
print(df[(df["Age"] > 22) | (df["CGPA"] > 3.5)])  # OR  -> matches all 5 here (everyone
                                                   #        satisfies at least one) --
                                                   #        notice OR is far less strict than AND
print(df[~(df["Age"] > 22)])                      # NOT -> Karim, Karim (the two 22-year-olds)

print(df[df["Age"].isin([22, 23])])               # -> Aisha(23), Karim(22), Karim(22)
print(df[df["Name"].str.contains("an")])          # substring match -> only "Farhan"
print(df.query("Age > 22 and CGPA > 3.5"))        # same result as the AND filter above
```

---

## 7. Missing Values

**Theory**

- Real data almost always has gaps (`NaN`). Before modeling you must either **remove** or **fill** them — most ML algorithms can't handle NaN.
- **Check:** `isnull()`/`isna()` (True = missing), `notnull()` (opposite)
- **Remove:** `dropna()` — drops whole rows (default) or columns (`axis=1`); `how="all"` only drops a row/col if **every** value is missing
- **Fill:** `fillna(value)` — replace NaN with a constant; `fillna(mean/median)` — replace with a column statistic; `ffill()`/`bfill()` — carry the previous/next valid value forward/back
- Choosing which strategy depends on how much data is missing and whether dropping rows would lose too much information.

> Recall: Farhan's CGPA was set to `np.nan` when we built `df`.

**Example**

```python
# --- Check ---
print(df.isnull())              # grid of True/False -> True where a value is missing
print(df.isnull().sum())        # -> CGPA: 1, every other column: 0
print(df.notnull())             # opposite of isnull()

# --- Remove ---
print(df.dropna())              # drops Farhan's row entirely -> 4 rows remain
print(df.dropna(axis=1))        # drops the whole CGPA column -> 4 columns remain
print(df.dropna(how="all"))     # nothing dropped -- no row is COMPLETELY empty

# --- Fill ---
print(df.fillna(0))                                  # Farhan's CGPA -> 0
print(df["CGPA"].fillna(df["CGPA"].mean()))          # Farhan's CGPA -> column mean (~3.725)
print(df["CGPA"].fillna(df["CGPA"].median()))        # Farhan's CGPA -> column median (3.8)
print(df["CGPA"].ffill())                            # Farhan's CGPA -> value above it (Karim's 3.8)
```

---

## 8. Duplicates

**Theory**

- Duplicate rows (identical values across all columns) can bias an ML model by over-weighting repeated samples.
- `duplicated()` → boolean Series, True on every repeat **after** the first occurrence (first copy is "clean")
- `drop_duplicates()` → removes those repeats, keeping the first occurrence by default

> Recall: row 4 (Karim, 22, 3.8, Male, 60000) is an exact copy of row 1.

**Example**

```python
print(df.duplicated())          # -> False,False,False,False,True   (row 4 flagged)
print(df.duplicated().sum())    # -> 1 duplicate row
print(df.drop_duplicates())     # -> row 4 removed, 4 rows remain (first copy kept)
```

---

## 9. Modifying Data

**Theory**

- Assigning to `df["new_col"]` either overwrites an existing column or creates a new one.
- `rename(columns={...})` → relabel columns without changing data
- `drop(labels, axis=)` → `axis=1` drops columns, `axis=0` drops rows
- `sort_values(col)` → reorder rows by a column's values
- `reset_index(drop=True)` → replace the current index with a fresh `0..n-1` RangeIndex (useful after filtering/dropping rows, since old labels get "holes" in them)
- Work on `df.copy()` when experimenting so the original `df` stays intact for later sections.

**Example**

```python
# (work on a copy so the shared `df` stays untouched)
demo = df.copy()

demo["Age"] = demo["Age"] + 1                          # everyone gets 1 year older
demo["Senior"] = demo["Age"] > 24                      # new boolean column
print(demo[["Name", "Age", "Senior"]])
# -> after the +1, only Farhan (26) and Nabila (25) end up Senior == True

print(demo.rename(columns={"CGPA": "GPA"}).columns)    # -> "GPA" replaces "CGPA"
print(demo.drop("Salary", axis=1).columns)             # -> "Salary" gone     (axis=1 = columns)
print(demo.drop(0, axis=0))                            # -> row label 0 (Aisha) removed (axis=0 = rows)
print(demo.sort_values("Age"))                         # -> the two Karims (youngest) come first
print(demo.sort_values("Age", ascending=False).head(1))# -> oldest student first -> Farhan
print(demo.reset_index(drop=True).index)               # -> RangeIndex(0..n-1), old labels discarded
```

---

## 10. Applying Functions (apply / map / lambda)

**Theory**

- `Series.apply(func)` → run `func` on every value of **one column**
- `Series.map(dict)` → replace each value using a lookup dict (great for simple manual encoding)
- `DataFrame.apply(func, axis=1)` → run `func` on every **row** (`axis=1` = "move across columns, once per row"); `axis=0` (default) runs once per **column** instead.
- Use these when a transformation isn't a built-in pandas method — e.g. custom bucketing/labeling logic.

**Example**

```python
# (fresh copy, original Age values: 23,22,25,24,22)
demo = df.copy()

print(demo["Age"].apply(lambda x: x * 2))              # -> every age doubled: 46,44,50,48,44

demo["Level"] = demo["Age"].apply(                     # build a new column with custom logic
    lambda x: "Senior" if x >= 24 else "Junior"
)
print(demo[["Name", "Age", "Level"]])
# -> Farhan (25) and Nabila (24) become "Senior", everyone else "Junior"

print(demo["Gender"].map({"Male": 0, "Female": 1}))    # -> Female -> 1, Male -> 0, row by row

print(demo.apply(lambda row: row["Age"] + row["CGPA"], axis=1))
# -> row-wise sum of Age + CGPA (axis=1 = "go across each row");
#    Farhan's result is NaN because his CGPA is missing
```

---

## 11. Grouping & Aggregation (groupby = split → apply → combine)

**Theory**

- `groupby(col)` splits the DataFrame into groups sharing the same value in `col`, applies an aggregation to each group, then combines the results into one summary table.
- Common aggregations: `mean`, `sum`, `min`, `max`, `size`/`count`.
- `.agg({...})` lets you apply **different** aggregations to different columns in a single call.
- This is the pandas equivalent of SQL's `GROUP BY`.

**Example**

```python
print(df.groupby("Gender")["Age"].mean())
# -> Female avg age = (23+24)/2 = 23.5  |  Male avg age = (22+25+22)/3 = 23.0

print(df.groupby("Gender").size())
# -> Female: 2 students, Male: 3 students

print(df.groupby("Gender").agg({                  # multiple aggregations at once
    "Age":  "mean",
    "CGPA": ["min", "max"]
}))
# -> one summary row per Gender, with every requested statistic as a column
```

---

## 12. Combining DataFrames

**Theory**

- `concat([...])` → stacks DataFrames that share the same columns, one on top of another (think: appending more rows of the same shape). Index labels are kept as-is unless `ignore_index=True`.
- `merge(a, b, on=key, how=...)` → SQL-style **JOIN** on a shared key column, combining columns from both tables side by side.
  - `how="inner"` → only keys present in **both** tables
  - `how="left"` → all keys from the left table; unmatched → `NaN`
  - (also `"right"` and `"outer"`, mirroring SQL joins)

**Example**

```python
# --- concat: stack tables that share the same columns ---
batch1 = pd.DataFrame({"Name": ["Aisha", "Karim"],  "Age": [23, 22]})
batch2 = pd.DataFrame({"Name": ["Farhan", "Nabila"], "Age": [25, 24]})

print(pd.concat([batch1, batch2]))                       # -> 4 rows, index repeats: 0,1,0,1
print(pd.concat([batch1, batch2], ignore_index=True))    # -> 4 rows, fresh index: 0,1,2,3

# --- merge: SQL-style join on a shared key column ---
students = pd.DataFrame({"ID": [1, 2, 3], "Name": ["Aisha", "Karim", "Farhan"]})
grades   = pd.DataFrame({"ID": [1, 2, 4], "CGPA": [3.4, 3.8, 3.6]})

print(pd.merge(students, grades, on="ID"))               # inner -> only IDs 1,2 (common to both)
print(pd.merge(students, grades, on="ID", how="left"))   # left  -> all of `students`; ID 3 -> CGPA NaN
```

---

## 13. Feature Selection (ML important)

**Theory**

- Every supervised ML model needs:
  - `X` = the input features (a DataFrame of predictor columns)
  - `y` = the target/label (a Series — the thing we want to predict)
- Selecting `X` means picking the columns that are actually useful predictors and dropping IDs, leakage columns, or the target itself.

**Example**

```python
X = df[["Age", "CGPA"]]   # features  (model input)
y = df["Salary"]          # target    (model output)

print(X.head())           # -> DataFrame of inputs
print(y.head())           # -> Series of the value we want to predict
```

---

## 14. Encoding Categorical Data

**Theory**

- ML models need **numbers**, not text, so categorical columns (like "Gender") must be converted before training.
- **Label Encoding** → maps each category to an integer (0, 1, 2, ...). Good for **binary or ordinal** categories (where order has meaning, e.g. Low/Medium/High). Can be done manually (`map`) or with sklearn's `LabelEncoder` (handles many categories automatically, assigns codes alphabetically).
- **One-Hot Encoding** → creates one new True/False column per category. Use for **nominal** categories (no real order, e.g. colors, cities) — label-encoding those would wrongly imply an order/magnitude. `drop_first=True` drops one of the new columns to avoid redundant/duplicate information (the "dummy variable trap").

**Example**

```python
encoded = df.copy()

# --- Label Encoding (manual) -- good for binary/ordinal categories ---
encoded["Gender_label"] = encoded["Gender"].map({"Male": 0, "Female": 1})
print(encoded[["Gender", "Gender_label"]])
# -> Female -> 1, Male -> 0

# --- Label Encoding (sklearn) -- handles many categories automatically ---
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded["Gender_sklearn"] = le.fit_transform(encoded["Gender"])
print(encoded[["Gender", "Gender_sklearn"]])
# -> sklearn assigns codes alphabetically: "Female" -> 0, "Male" -> 1

# --- One-Hot Encoding -- for nominal categories (no real order) ---
print(pd.get_dummies(df, columns=["Gender"]).filter(like="Gender"))
# -> creates Gender_Female / Gender_Male columns (one "hot" True per row)
print(pd.get_dummies(df, columns=["Gender"], drop_first=True).filter(like="Gender"))
# -> drops Gender_Female, keeps only Gender_Male (avoids redundant info)
```

---

## 15. Train-Test Split (with sklearn)

**Theory**

- You never train AND evaluate on the same data — that overstates how good the model really is (it can just "memorize").
- `train_test_split(X, y, test_size=...)` randomly splits rows into a **training set** (to fit the model) and a **test set** (to evaluate it on unseen data).
- `random_state` fixes the random shuffle so the split is reproducible across runs.

**Example**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42     # random_state -> reproducible split
)
print("Train size:", X_train.shape, " Test size:", X_test.shape)
# -> with 5 rows and test_size=0.2: 4 rows go to train, 1 row goes to test
```

---

## 16. Feature Scaling

**Theory**

- Many ML algorithms (distance-based, gradient-based) perform badly if features have very different scales (e.g. Age ~20-30 vs Salary ~50000+) — the bigger-magnitude feature dominates.
- **Standardization** (`StandardScaler`) → rescales to mean=0, std=1. Good default; assumes roughly normal-ish distribution.
- **Normalization** (`MinMaxScaler`) → squashes values into `[0, 1]` based on min/max. No distribution assumption, but sensitive to outliers (a single extreme value skews the whole scale).
- Always fit the scaler on **train** data only, then transform both train and test with it (shown simply here on the full set for clarity).

**Example**

```python
# Use Age & Salary here (no missing values) so scaling has clean numeric input.
X_num = df[["Age", "Salary"]]

# --- Standardization -- mean = 0, std = 1 ---
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit_transform(X_num))
# -> every column is re-centred around 0 with unit variance

# --- Normalization -- squashes values into [0, 1] ---
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit_transform(X_num))
# -> the smallest value in each column becomes 0, the largest becomes 1
```

---

## 17. ML Workflow Summary

```
Raw Data
  |
Pandas Cleaning          (sections 7-8: missing values, duplicates)
  |
Feature Selection (X, y) (section 13)
  |
Encoding                 (section 14: categorical -> numeric)
  |
Scaling                  (section 16: standardize / normalize)
  |
Train/Test Split         (section 15)
  |
Model Training
```

> Note: train/test split usually happens **before** fitting scalers/encoders in a real pipeline (fit on train only) — the linear order above is simplified for memorization, not a strict execution order.

---

## 18. Quick Memory Tricks

| Concept | Meaning |
|---|---|
| `Series` | 1 column |
| `DataFrame` | table (rows + columns) |
| `iloc` | select by **position** (integers, end excluded) |
| `loc` | select by **label** (names, end included) |
| `[]` | select column(s) |
| `df[df["x"] > value]` | filter rows with a boolean mask |
| `groupby` | split → apply → combine |
| `X` | input features |
| `y` | output / target |
