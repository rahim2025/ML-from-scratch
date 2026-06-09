"""
SCIKIT-LEARN REVISION MATERIALS
=================================
Comprehensive guide covering Phases 1-3 of scikit-learn fundamentals

Phase 1: Foundations
Phase 2: Data Preparation  
Phase 3: Core Algorithms
"""

# ============================================================================
# PHASE 1: FOUNDATIONS
# ============================================================================

"""
WHAT IS SCIKIT-LEARN?
- Python library for machine learning built on NumPy and SciPy
- Provides simple and efficient tools for data analysis and modeling
- Fits in the ecosystem: NumPy (arrays) → Scikit-learn (ML) → Pandas (data)
- Part of larger ML stack: Scikit-learn + TensorFlow/PyTorch + XGBoost
"""

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


# SECTION 1.1: THE ESTIMATOR API
# =================================
"""
The Estimator API is the heart of scikit-learn. Every model follows this pattern:

1. Create/instantiate the model with hyperparameters
2. fit() - Learn patterns from training data
3. predict() - Make predictions on new data
4. transform() / fit_transform() - Apply transformations (for scalers, imputers)

Benefits:
- Consistent interface across all algorithms
- Easy to switch between different models
- Pipeline-friendly
"""

# Example: Linear Regression (Estimator without transform)
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 1: Instantiate
model = LinearRegression()

# Step 2: Fit (learn patterns)
model.fit(X_train, y_train)

# Step 3: Predict
y_pred = model.predict(X_test)
print(f"Model MSE: {mean_squared_error(y_test, y_pred):.4f}")


# Example: StandardScaler (Transformer with fit, transform, fit_transform)
scaler = StandardScaler()

# Method 1: Separate fit and transform
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Method 2: Fit and transform in one call (only on training data!)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use SAME scaler on test data

print(f"Original mean: {X_train.mean(axis=0)[0]:.4f}")
print(f"Scaled mean: {X_train_scaled.mean(axis=0)[0]:.6f}")


# SECTION 1.2: TRAILING UNDERSCORE CONVENTION
# =============================================
"""
In scikit-learn:
- Attributes WITHOUT trailing underscore: Set by user (hyperparameters)
- Attributes WITH trailing underscore (_): Set during fit() (learned parameters)

This convention helps you distinguish between:
- Input parameters: model.max_depth = 5
- Learned parameters: model.feature_importances_, model.coef_, model.classes_
"""

# Hyperparameters (set by user, no trailing underscore)
dt = DecisionTreeClassifier(max_depth=5)

# After fitting, learned attributes appear (with trailing underscore)
dt.fit(X_train, y_train)

print(f"Max depth (hyperparameter): {dt.max_depth}")
print(f"Feature importances (learned): {dt.feature_importances_}")  # Trailing underscore!
print(f"Classes (learned): {dt.classes_}")  # Trailing underscore!

# More examples of learned attributes
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print(f"Coefficients shape: {lr.coef_.shape}")  # (n_classes, n_features)
print(f"Intercept: {lr.intercept_}")
