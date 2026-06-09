"""
====================================================================
                  PANDAS REVISION SHEET (for ML)
====================================================================
Each topic below follows the same pattern:

    THEORY   -> what it is / why it matters   (comments)
    EXAMPLE  -> runnable code that prints real output

Most examples reuse ONE shared sample dataset called `df`
(created in section 2), so you can watch the same data flow
through inspection -> cleaning -> feature selection -> encoding
-> scaling, just like a real ML pipeline.

Run this file directly to see every example's output for real:
    python panda.py

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

import numpy as np
import pandas as pd


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

# --- Example ---
print(pd.DataFrame({"Age": [23, 22], "CGPA": [3.4, 3.8]}))
# Output:
#    Age  CGPA
# 0   23   3.4
# 1   22   3.8
#
# Each ROW    = one student   (a "sample")
# Each COLUMN = one attribute (a "feature") about that student


# ====================================================================
# 2. CORE DATA STRUCTURES
# ====================================================================

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

# Other common ways to build / load a DataFrame
print(pd.DataFrame([[23, 3.4], [22, 3.8]], columns=["Age", "CGPA"]))
# pd.read_csv("data.csv")          # from a CSV file   (shown for syntax -- no file here)
# pd.read_excel("data.xlsx")       # from an Excel file
# pd.read_json("data.json")        # from a JSON file

# Saving a DataFrame back out
# df.to_csv("out.csv", index=False)
# df.to_excel("out.xlsx", index=False)


# ====================================================================
# 3. BASIC INSPECTION
# ====================================================================
# Usually the FIRST commands you run on new data: size, structure,
# data types, and a quick statistical summary.

# --- Example ---
print(df.head(3))      # first 3 rows           -> Aisha, Karim, Farhan
print(df.tail(2))      # last 2 rows            -> Nabila, Karim
print(df.shape)        # (n_rows, n_columns)    -> (5, 5)
print(df.columns)      # column names           -> Name, Age, CGPA, Gender, Salary
df.info()              # dtypes + non-null counts + memory (prints directly)
print(df.describe())   # stats for numeric cols -> count, mean, std, min, max...

print(df["Age"].unique())         # -> [23 22 25 24]            (4 distinct ages)
print(df["Age"].nunique())        # -> 4
print(df["Age"].value_counts())   # -> 22 appears twice, the rest appear once


# ====================================================================
# 4. SELECTING COLUMNS
# ====================================================================
# --- Example ---
print(df["Age"])                   # single column  -> Series of 5 ages
print(df[["Age"]])                 # single column  -> DataFrame (note double brackets)
print(df[["Age", "CGPA"]])         # multiple columns -> 2-column DataFrame
print(df.Age)                      # attribute-style access (only if name has no spaces)


# ====================================================================
# 5. SELECTING ROWS  (loc vs iloc vs at/iat)
# ====================================================================
# --- Example: iloc -> integer POSITION based (end EXCLUDED, like list slicing) ---
print(df.iloc[0])             # first row              -> Aisha
print(df.iloc[0:2])           # rows at positions 0,1  -> Aisha, Karim   (2 rows)
print(df.iloc[0, 1])          # row-pos 0, col-pos 1   -> 23 (Aisha's Age)
print(df.iloc[:, 0])          # all rows, first column -> the Name column

# --- Example: loc -> LABEL based (uses index/column names, end INCLUDED) ---
print(df.loc[0])                     # row with index label 0 -> Aisha
print(df.loc[0:2])                   # labels 0,1,2 -> Aisha, Karim, Farhan (3 rows!)
print(df.loc[0, "Age"])              # row label 0, column "Age" -> 23
print(df.loc[:, ["Age", "CGPA"]])    # all rows, selected columns by name

# --- Example: at / iat -> fast access to a SINGLE scalar value ---
print(df.at[0, "Age"])        # label based, single value    -> 23
print(df.iat[0, 0])           # position based, single value -> "Aisha"

# GOTCHA: iloc[0:2] returns 2 rows (end excluded) but loc[0:2] returns
#         3 rows (end included) -- this trips almost everyone up at least once!


# ====================================================================
# 6. FILTERING DATA  (boolean masks)
# ====================================================================
# --- Example ---
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


# ====================================================================
# 7. MISSING VALUES
# ====================================================================
# Recall: Farhan's CGPA was set to np.nan when we built `df`.

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


# ====================================================================
# 8. DUPLICATES
# ====================================================================
# Recall: row 4 (Karim, 22, 3.8, Male, 60000) is an exact copy of row 1.

# --- Example ---
print(df.duplicated())          # -> False,False,False,False,True   (row 4 flagged)
print(df.duplicated().sum())    # -> 1 duplicate row
print(df.drop_duplicates())     # -> row 4 removed, 4 rows remain (first copy kept)


# ====================================================================
# 9. MODIFYING DATA
# ====================================================================
# --- Example (work on a copy so the shared `df` stays untouched) ---
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


# ====================================================================
# 10. APPLYING FUNCTIONS  (apply / map / lambda)
# ====================================================================
# --- Example (fresh copy, original Age values: 23,22,25,24,22) ---
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


# ====================================================================
# 11. GROUPING & AGGREGATION  (groupby = split -> apply -> combine)
# ====================================================================
# --- Example ---
print(df.groupby("Gender")["Age"].mean())
# -> Female avg age = (23+24)/2 = 23.5  |  Male avg age = (22+25+22)/3 = 23.0

print(df.groupby("Gender").size())
# -> Female: 2 students, Male: 3 students

print(df.groupby("Gender").agg({                  # multiple aggregations at once
    "Age":  "mean",
    "CGPA": ["min", "max"]
}))
# -> one summary row per Gender, with every requested statistic as a column


# ====================================================================
# 12. COMBINING DATAFRAMES
# ====================================================================
# --- Example: concat -> stack tables that share the same columns ---
batch1 = pd.DataFrame({"Name": ["Aisha", "Karim"],  "Age": [23, 22]})
batch2 = pd.DataFrame({"Name": ["Farhan", "Nabila"], "Age": [25, 24]})

print(pd.concat([batch1, batch2]))                       # -> 4 rows, index repeats: 0,1,0,1
print(pd.concat([batch1, batch2], ignore_index=True))    # -> 4 rows, fresh index: 0,1,2,3

# --- Example: merge -> SQL-style join on a shared key column ---
students = pd.DataFrame({"ID": [1, 2, 3], "Name": ["Aisha", "Karim", "Farhan"]})
grades   = pd.DataFrame({"ID": [1, 2, 4], "CGPA": [3.4, 3.8, 3.6]})

print(pd.merge(students, grades, on="ID"))               # inner -> only IDs 1,2 (common to both)
print(pd.merge(students, grades, on="ID", how="left"))   # left  -> all of `students`; ID 3 -> CGPA NaN


# ====================================================================
# 13. FEATURE SELECTION  (ML important)
# ====================================================================
# --- Example ---
X = df[["Age", "CGPA"]]   # features  (model input)
y = df["Salary"]          # target    (model output)

print(X.head())           # -> DataFrame of inputs
print(y.head())           # -> Series of the value we want to predict


# ====================================================================
# 14. ENCODING CATEGORICAL DATA
# ====================================================================
encoded = df.copy()

# --- Example: Label Encoding (manual) -- good for binary/ordinal categories ---
encoded["Gender_label"] = encoded["Gender"].map({"Male": 0, "Female": 1})
print(encoded[["Gender", "Gender_label"]])
# -> Female -> 1, Male -> 0

# --- Example: Label Encoding (sklearn) -- handles many categories automatically ---
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded["Gender_sklearn"] = le.fit_transform(encoded["Gender"])
print(encoded[["Gender", "Gender_sklearn"]])
# -> sklearn assigns codes alphabetically: "Female" -> 0, "Male" -> 1

# --- Example: One-Hot Encoding -- for nominal categories (no real order) ---
print(pd.get_dummies(df, columns=["Gender"]).filter(like="Gender"))
# -> creates Gender_Female / Gender_Male columns (one "hot" True per row)
print(pd.get_dummies(df, columns=["Gender"], drop_first=True).filter(like="Gender"))
# -> drops Gender_Female, keeps only Gender_Male (avoids redundant info)


# ====================================================================
# 15. TRAIN-TEST SPLIT  (with sklearn)
# ====================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42     # random_state -> reproducible split
)
print("Train size:", X_train.shape, " Test size:", X_test.shape)
# -> with 5 rows and test_size=0.2: 4 rows go to train, 1 row goes to test


# ====================================================================
# 16. FEATURE SCALING
# ====================================================================
# Use Age & Salary here (no missing values) so scaling has clean numeric input.
X_num = df[["Age", "Salary"]]

# --- Example: Standardization -- mean = 0, std = 1 ---
# (good when the data is roughly normally distributed)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit_transform(X_num))
# -> every column is re-centred around 0 with unit variance

# --- Example: Normalization -- squashes values into [0, 1] ---
# (no assumption about the distribution of the data)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit_transform(X_num))
# -> the smallest value in each column becomes 0, the largest becomes 1


# ====================================================================
# 17. ML WORKFLOW SUMMARY
# ====================================================================
#   Raw Data
#     |
#   Pandas Cleaning          (sections 7-8: missing values, duplicates)
#     |
#   Feature Selection (X, y) (section 13)
#     |
#   Encoding                 (section 14: categorical -> numeric)
#     |
#   Scaling                  (section 16: standardize / normalize)
#     |
#   Train/Test Split         (section 15)
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
