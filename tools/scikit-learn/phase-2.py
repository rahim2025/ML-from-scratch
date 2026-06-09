from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    classification_report, confusion_matrix
)
import numpy as np
import pandas as pd

# ============================================================================
# PHASE 2: DATA PREPARATION
# ============================================================================

"""
The quality of your data determines the quality of your model.
"Garbage in, garbage out" — proper data prep is crucial.
"""

# SECTION 2.1: LOADING BUILT-IN DATASETS
# ========================================

# Load Iris (classification)
iris = load_iris()
print(f"Iris data shape: {iris.data.shape}")  # (150, 4)
print(f"Iris targets: {iris.target}")  # Classes: 0, 1, 2
print(f"Iris feature names: {iris.feature_names}")

# Load Diabetes (regression)
diabetes = load_diabetes()
print(f"Diabetes data shape: {diabetes.data.shape}")  # (442, 10)
print(f"Diabetes target shape: {diabetes.target.shape}")  # Continuous values


# SECTION 2.2: EXPLORING DATA WITH PANDAS
# =========================================

# Convert to DataFrame for easier exploration
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

print("=" * 60)
print("DATA EXPLORATION WITH PANDAS")
print("=" * 60)

# Shape: dimensions of the data
print(f"Shape: {iris_df.shape}")  # (rows, columns)

# Check for missing values
print(f"Missing values:\n{iris_df.isnull().sum()}")

# Statistical summary
print(f"\nDescribe:\n{iris_df.describe()}")

# Head and tail
print(f"\nFirst few rows:\n{iris_df.head()}")

# Data types
print(f"\nData types:\n{iris_df.dtypes}")

# Value counts for categorical
print(f"\nTarget distribution:\n{iris_df['target'].value_counts()}")


# SECTION 2.3: TRAIN/TEST SPLIT
# ==============================
"""
CRITICAL: Always split BEFORE any data preparation!
Why? Prevent data leakage - information from test set leaks into training

Correct order:
1. Train/Test Split
2. Fit scaler/imputer on TRAINING data only
3. Transform BOTH train and test with the fitted scaler/imputer

Wrong: Fit scaler on entire dataset → Split → Train model (LEAKS INFO!)
"""

X = iris.data
y = iris.target

# Standard train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training target distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Test target distribution:\n{pd.Series(y_test).value_counts()}")

# random_state: reproducibility (use any fixed number)
# stratify: ensures same class distribution in train/test (important for classification)


# SECTION 2.4: FEATURE SCALING
# =============================
"""
Why scale?
- Some algorithms (KNN, SVM, Neural Networks) are distance-based
- Features with large ranges dominate those with small ranges
- Gradient descent converges faster on scaled features

When NOT to scale?
- Tree-based models (Decision Tree, Random Forest) - scale-invariant
- Linear/Logistic Regression with regularization - usually scale anyway

Scaling types:
1. StandardScaler: (X - mean) / std → mean=0, std=1
2. MinMaxScaler: (X - min) / (max - min) → range [0, 1]
3. RobustScaler: (X - median) / IQR → handles outliers better
"""

print("\n" + "=" * 60)
print("FEATURE SCALING EXAMPLE")
print("=" * 60)

# Original data statistics
print(f"Original data - Min: {X_train.min(axis=0)[0]:.2f}, Max: {X_train.max(axis=0)[0]:.2f}")
print(f"Original data - Mean: {X_train.mean(axis=0)[0]:.2f}, Std: {X_train.std(axis=0)[0]:.2f}")

# StandardScaler
scaler_standard = StandardScaler()
X_train_std = scaler_standard.fit_transform(X_train)
X_test_std = scaler_standard.transform(X_test)  # Use SAME scaler!

print(f"\nStandardScaler - Mean: {X_train_std.mean(axis=0)[0]:.6f}, Std: {X_train_std.std(axis=0)[0]:.2f}")

# MinMaxScaler
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

print(f"MinMaxScaler - Min: {X_train_minmax.min(axis=0)[0]:.2f}, Max: {X_train_minmax.max(axis=0)[0]:.2f}")


# SECTION 2.5: HANDLING MISSING VALUES
# =====================================
"""
Missing data strategies:
1. Delete: Remove rows/columns with missing values (loses data)
2. Impute: Fill missing values with a statistic (mean, median, most frequent)
3. Advanced: Use models to predict missing values

SimpleImputer is the standard tool for basic imputation
"""

# Create data with missing values
data_with_missing = X_train.copy()
data_with_missing[0, 0] = np.nan
data_with_missing[1, 1] = np.nan
data_with_missing[2, 2] = np.nan

print(f"\nMissing values count: {np.isnan(data_with_missing).sum()}")

# Impute with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_with_missing)

print(f"After imputation, missing values: {np.isnan(data_imputed).sum()}")

# Common strategies:
# - 'mean': average of non-missing values
# - 'median': middle value (robust to outliers)
# - 'most_frequent': mode (for categorical)
# - 'constant': fill with a specific value


# SECTION 2.6: DATA LEAKAGE
# ==========================
"""
Data leakage = test set information influencing the model

Example 1: Scaling leak
WRONG:
  scaler.fit(X_combined)        # Fit on full dataset
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)

RIGHT:
  scaler.fit(X_train)           # Fit ONLY on training data
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)

Example 2: Feature selection leak
WRONG:
  features = select_features(X_full, y_full)  # Using test data!
  X_train = X_full[features]
  X_test = X_full[features]

RIGHT:
  features = select_features(X_train, y_train)  # Using ONLY train data
  X_train = X_train[features]
  X_test = X_test[features]

Impact: Inflated test scores that don't reflect real performance
"""

# Demonstrate leakage
print("\n" + "=" * 60)
print("DATA LEAKAGE DEMONSTRATION")
print("=" * 60)

# Correct way (no leakage)
scaler_correct = StandardScaler()
scaler_correct.fit(X_train)
X_train_correct = scaler_correct.transform(X_train)
X_test_correct = scaler_correct.transform(X_test)

# Wrong way (leakage - fit on test data!)
scaler_wrong = StandardScaler()
scaler_wrong.fit(X_test)  # BUG: Fitting on test data!
X_test_wrong = scaler_wrong.transform(X_test)

print(f"Correct scaling - test mean: {X_test_correct.mean(axis=0)[0]:.6f}")
print(f"Leaky scaling - test mean: {X_test_wrong.mean(axis=0)[0]:.6f}")
print("(Different means indicate the leaky version is biased toward test data)")


# ===========================================