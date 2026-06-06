"""
====================================================================
                  PANDAS REVISION SHEET (for ML)
====================================================================
Quick-reference notes for revising Pandas. Read top to bottom,
or jump to a section using the table of contents below.

TABLE OF CONTENTS
  1.  What is Pandas?
  2.  Core Data Structures (Series, DataFrame)
  3.  Basic Inspection
  4.  Selecting Columns
  5.  Selecting Rows (loc vs iloc vs at/iat)
  6.  Filtering Data (boolean masks, query)
  7.  Missing Values
  8.  Duplicates
  9.  Modifying Data (rename, drop, sort, new columns)
  10. Applying Functions (apply / map / lambda)
  11. Grouping & Aggregation (groupby)
  12. Combining DataFrames (concat / merge)
  13. Feature Selection (X, y)
  14. Encoding Categorical Data
  15. Train-Test Split
  16. Feature Scaling
  17. ML Workflow Summary
  18. Quick Memory Tricks
====================================================================
"""

import pandas as pd
import numpy as np


# ====================================================================
# 1. WHAT IS PANDAS?
# ====================================================================
# Pandas = Python library for working with tabular/structured data
#          (think: Excel sheets or SQL tables, but in Python).
#
# Terminology:
#   Row    = sample / observation / record
#   Column = feature / variable / attribute
#
# Typical uses in an ML pipeline:
#   - Loading data       (csv, excel, sql, json, ...)
#   - Cleaning data      (missing values, duplicates, wrong types)
#   - Exploring data     (statistics, grouping, sorting)
#   - Engineering/selecting features
#   - Producing the final X, y arrays that feed into a model


# ====================================================================
# 2. CORE DATA STRUCTURES
# ====================================================================

# --- Series (1D): a single labeled column / array of values ---
s = pd.Series([10, 20, 30])
s = pd.Series([10, 20, 30], index=["a", "b", "c"])      # custom index

# --- DataFrame (2D): a table of rows & columns (the main structure) ---
df = pd.DataFrame({
    "Age":  [23, 22, 25],
    "CGPA": [3.4, 3.8, 3.6]
})

# Other common ways to build / load a DataFrame
df = pd.DataFrame([[23, 3.4], [22, 3.8]], columns=["Age", "CGPA"])
df = pd.read_csv("data.csv")          # from a CSV file
df = pd.read_excel("data.xlsx")       # from an Excel file
df = pd.read_json("data.json")        # from a JSON file

# Saving a DataFrame back out
df.to_csv("out.csv", index=False)
df.to_excel("out.xlsx", index=False)


# ====================================================================
# 3. BASIC INSPECTION
# ====================================================================
df.head()        # first 5 rows
df.head(10)      # first 10 rows
df.tail()        # last 5 rows
df.shape         # (n_rows, n_columns)
df.columns       # column names
df.index         # row index
df.dtypes        # data type of each column
df.info()        # dtypes + non-null counts + memory usage
df.describe()    # stats for numeric columns: count, mean, std, min, max...

df["Age"].unique()         # array of unique values in a column
df["Age"].nunique()        # number of unique values
df["Age"].value_counts()   # frequency count of each distinct value


# ====================================================================
# 4. SELECTING COLUMNS
# ====================================================================
df["Age"]                   # single column  -> Series
df[["Age"]]                 # single column  -> DataFrame (note double brackets)
df[["Age", "CGPA"]]         # multiple columns -> DataFrame
df.Age                      # attribute-style access (only if name has no spaces)


# ====================================================================
# 5. SELECTING ROWS  (loc vs iloc vs at/iat)
# ====================================================================
# iloc -> integer POSITION based (like Python list slicing, end EXCLUDED)
df.iloc[0]            # first row
df.iloc[1]            # second row
df.iloc[0:2]          # rows at positions 0 and 1
df.iloc[0, 1]         # value at row-position 0, column-position 1
df.iloc[:, 0]         # all rows, first column

# loc -> LABEL based (uses index/column names, end INCLUDED)
df.loc[0]                    # row with index label 0
df.loc[0:2]                  # rows with labels 0,1,2  (end included!)
df.loc[0, "Age"]             # row label 0, column "Age"
df.loc[:, ["Age", "CGPA"]]   # all rows, selected columns by name

# at / iat -> fast access to a SINGLE scalar value (label / position)
df.at[0, "Age"]       # label based, single value
df.iat[0, 0]          # position based, single value


# ====================================================================
# 6. FILTERING DATA  (boolean masks)
# ====================================================================
df[df["Age"] > 22]                            # rows where Age > 22
df[df["CGPA"] > 3.5]

# Combine conditions: use & (and), | (or), ~ (not) -- always with parentheses!
df[(df["Age"] > 22) & (df["CGPA"] > 3.5)]     # AND
df[(df["Age"] > 22) | (df["CGPA"] > 3.5)]     # OR
df[~(df["Age"] > 22)]                         # NOT

df[df["Age"].isin([22, 23])]                  # value is one of a list
df[df["Name"].str.contains("ana")]            # substring match (string columns)
df.query("Age > 22 and CGPA > 3.5")           # same filter, more readable syntax


# ====================================================================
# 7. MISSING VALUES
# ====================================================================
# --- Check ---
df.isnull()              # grid of True/False -> True where value is missing
df.isnull().sum()        # count of missing values PER COLUMN
df.notnull()             # opposite of isnull()

# --- Remove ---
df.dropna()              # drop rows that have ANY missing value
df.dropna(axis=1)        # drop columns that have ANY missing value
df.dropna(how="all")     # drop rows where ALL values are missing

# --- Fill ---
df.fillna(0)                                  # replace every missing value with 0
df["Age"].fillna(df["Age"].mean())            # fill with the column's mean
df["Age"].fillna(df["Age"].median())          # fill with the column's median
df["Age"].fillna(method="ffill")              # forward-fill (carry previous value down)


# ====================================================================
# 8. DUPLICATES
# ====================================================================
df.duplicated()          # True for rows that are duplicates of an earlier row
df.duplicated().sum()    # how many duplicate rows exist
df.drop_duplicates()     # remove duplicate rows (keeps the first occurrence)


# ====================================================================
# 9. MODIFYING DATA
# ====================================================================
df["Age"] = df["Age"] + 1                         # overwrite/transform a column
df["Senior"] = df["Age"] > 23                     # create a new column from a condition
df.rename(columns={"CGPA": "GPA"})                # rename column(s)
df.drop("CGPA", axis=1)                           # drop a column      (axis=1)
df.drop(0, axis=0)                                # drop a row by index label (axis=0)
df.sort_values("Age")                             # sort ascending by a column
df.sort_values("Age", ascending=False)            # sort descending
df.reset_index(drop=True)                         # renumber index as 0..n-1


# ====================================================================
# 10. APPLYING FUNCTIONS  (apply / map / lambda)
# ====================================================================
df["Age"].apply(lambda x: x * 2)                  # apply a function to each value

df["Category"] = df["Age"].apply(                 # build a new column with custom logic
    lambda x: "Adult" if x >= 18 else "Minor"
)

df["Gender"].map({"Male": 0, "Female": 1})        # map/replace values using a dict

df.apply(lambda row: row["Age"] + row["CGPA"], axis=1)   # row-wise operation (axis=1)


# ====================================================================
# 11. GROUPING & AGGREGATION  (groupby = split -> apply -> combine)
# ====================================================================
df.groupby("Gender")["Age"].mean()        # average Age for each Gender group
df.groupby("Gender").size()               # number of rows in each group
df.groupby("Gender").agg({                # multiple aggregations at once
    "Age":  "mean",
    "CGPA": ["min", "max"]
})


# ====================================================================
# 12. COMBINING DATAFRAMES
# ====================================================================
pd.concat([df, df])                          # stack DataFrames vertically (rows)
pd.concat([df, df], axis=1)                  # stack DataFrames side by side (columns)

pd.merge(df, df, on="Age")                   # SQL-style join on a shared column
pd.merge(df, df, on="Age", how="left")       # join type: left / right / inner / outer


# ====================================================================
# 13. FEATURE SELECTION  (ML important)
# ====================================================================
X = df[["Age", "CGPA"]]   # features  (model input)
y = df["Age"]             # target    (model output) -- replace with your real label column


# ====================================================================
# 14. ENCODING CATEGORICAL DATA
# ====================================================================
# --- Label Encoding (manual): good for binary / ordinal categories ---
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# --- Label Encoding (sklearn): handles many categories automatically ---
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# --- One-Hot Encoding: for nominal categories (no real order between them) ---
pd.get_dummies(df, columns=["Gender"])
pd.get_dummies(df, columns=["Gender"], drop_first=True)   # drop first col to avoid redundancy


# ====================================================================
# 15. TRAIN-TEST SPLIT  (with sklearn)
# ====================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42     # random_state -> reproducible split
)


# ====================================================================
# 16. FEATURE SCALING
# ====================================================================
# --- Standardization: mean = 0, std = 1 (good when data is ~normally distributed) ---
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Normalization: squashes values into [0, 1] (no distribution assumption) ---
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# ====================================================================
# 17. ML WORKFLOW SUMMARY
# ====================================================================
#   Raw Data
#     |
#   Pandas Cleaning          (missing values, duplicates, dtypes)
#     |
#   Feature Selection (X, y)
#     |
#   Encoding                 (categorical -> numeric)
#     |
#   Scaling                  (standardize / normalize numeric features)
#     |
#   Train/Test Split
#     |
#   Model Training


# ====================================================================
# 18. QUICK MEMORY TRICKS
# ====================================================================
#   Series                = 1 column
#   DataFrame             = table (rows + columns)
#   iloc                  = select by POSITION  (integers, end excluded)
#   loc                   = select by LABEL     (names, end included)
#   []                    = select column(s)
#   df[df["x"] > value]   = filter rows with a boolean mask
#   groupby               = split -> apply -> combine
#   X                     = input features
#   y                     = output / target
