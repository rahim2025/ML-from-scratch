
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

# PHASE 3: CORE ALGORITHMS
# ============================================================================

"""
Master the 5 core algorithms:
1. Linear Regression - continuous predictions
2. Logistic Regression - binary/multiclass classification
3. Decision Tree - interpretable tree-based model
4. K-Nearest Neighbors - instance-based learning
5. Random Forest - ensemble of trees
"""

# Use scaled data for distance-based algorithms
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)


# SECTION 3.1: LINEAR REGRESSION
# ===============================
"""
Use for: Predicting continuous values (price, temperature, demand)
Output: Real numbers (not bounded)
Loss function: Mean Squared Error (MSE)
Evaluation metrics:
  - MSE: Average squared difference (lower is better)
  - RMSE: √MSE (same units as target)
  - MAE: Mean Absolute Error
  - R²: Coefficient of determination (0-1, higher is better)
          - 1.0 = perfect prediction
          - 0.0 = model as good as predicting mean
          - negative = worse than predicting mean

Formula:
  y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
  (where b's are learned coefficients)
"""

print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

# Load regression dataset
diabetes = load_diabetes()
X_reg = diabetes.data
y_reg = diabetes.target

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Create and train model
lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = lr_model.predict(X_test_reg)

# Evaluate
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Coefficients shape: {lr_model.coef_.shape}")
print(f"Intercept: {lr_model.intercept_:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Interpretation:
# - If R² = 0.85, the model explains 85% of variance in the data
# - Coefficients show how much each feature affects predictions


# SECTION 3.2: LOGISTIC REGRESSION
# =================================
"""
Use for: Binary or multiclass classification
Output: Probability (0-1 for binary, probabilities summing to 1 for multiclass)
Loss function: Log Loss (cross-entropy)
Evaluation metrics:
  - Accuracy: (TP + TN) / Total (not good for imbalanced data)
  - Precision: TP / (TP + FP) - "Of predicted positives, how many are correct?"
  - Recall: TP / (TP + FN) - "Of actual positives, how many did we find?"
  - F1-Score: Harmonic mean of precision and recall
  - Confusion Matrix: Shows TP, FP, TN, FN

Formula:
  p(y=1) = 1 / (1 + e^(-z))  where z = b₀ + b₁x₁ + ...
  (sigmoid function squashes output to [0, 1])
"""

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

# Binary classification example
iris_binary = load_iris()
# Create binary classification: class 0 vs others
X_binary = iris_binary.data
y_binary = (iris_binary.target == 0).astype(int)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

# Create and train
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_bin, y_train_bin)

# Predictions
y_pred_log = logistic_model.predict(X_test_bin)
y_proba = logistic_model.predict_proba(X_test_bin)  # Probabilities for both classes

# Evaluate
accuracy = accuracy_score(y_test_bin, y_pred_log)
conf_matrix = confusion_matrix(y_test_bin, y_pred_log)
class_report = classification_report(y_test_bin, y_pred_log)

print(f"Accuracy: {accuracy:.4f}")
print(f"\nConfusion Matrix:\n{conf_matrix}")
print(f"(Row 0: true negatives={conf_matrix[0,0]}, false positives={conf_matrix[0,1]})")
print(f"(Row 1: false negatives={conf_matrix[1,0]}, true positives={conf_matrix[1,1]})")
print(f"\nClassification Report:\n{class_report}")
print(f"Probability predictions (first 5):\n{y_proba[:5]}")


# SECTION 3.3: DECISION TREE
# ===========================
"""
Use for: Classification or regression, interpretable models
Output: Class (classification) or continuous value (regression)
Key concepts:
  - Splits data recursively based on feature values
  - Each split maximizes information gain (or minimizes impurity)
  - Creates a tree that mirrors human decision-making

Hyperparameters:
  - max_depth: Maximum depth of tree (lower = simpler, less overfit)
  - min_samples_split: Minimum samples to split a node
  - min_samples_leaf: Minimum samples in leaf node
  - criterion: 'gini' (classification) or 'squared_error' (regression)

Advantages:
  - Interpretable: Can visualize and explain decisions
  - Handles non-linear relationships
  - No scaling needed
  - Feature importance automatically computed

Disadvantages:
  - Prone to overfitting
  - Unstable: small data changes → big tree changes
"""

print("\n" + "=" * 60)
print("DECISION TREE")
print("=" * 60)

# Multiclass classification with Decision Tree
iris_data = load_iris()
X_tree = iris_data.data
y_tree = iris_data.target

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_tree, y_tree, test_size=0.2, random_state=42
)

# Create tree (NO scaling needed for trees!)
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train_tree, y_train_tree)

# Predictions
y_pred_tree = dt_model.predict(X_test_tree)

# Evaluate
accuracy_tree = accuracy_score(y_test_tree, y_pred_tree)
feature_importance = dt_model.feature_importances_

print(f"Accuracy: {accuracy_tree:.4f}")
print(f"\nFeature Importances:")
for name, importance in zip(iris_data.feature_names, feature_importance):
    print(f"  {name}: {importance:.4f}")
print(f"(Sum should equal 1.0: {feature_importance.sum():.4f})")

# Overfitting demonstration
print("\nOverfitting example:")
dt_shallow = DecisionTreeClassifier(max_depth=2, random_state=42)
dt_deep = DecisionTreeClassifier(max_depth=10, random_state=42)

dt_shallow.fit(X_train_tree, y_train_tree)
dt_deep.fit(X_train_tree, y_train_tree)

print(f"Shallow tree (max_depth=2):")
print(f"  Train accuracy: {dt_shallow.score(X_train_tree, y_train_tree):.4f}")
print(f"  Test accuracy: {dt_shallow.score(X_test_tree, y_test_tree):.4f}")

print(f"Deep tree (max_depth=10):")
print(f"  Train accuracy: {dt_deep.score(X_train_tree, y_train_tree):.4f}")
print(f"  Test accuracy: {dt_deep.score(X_test_tree, y_test_tree):.4f}")
print("(Note: Deep tree overfits - high train, lower test)")


# SECTION 3.4: K-NEAREST NEIGHBORS (KNN)
# ========================================
"""
Use for: Classification or regression
Algorithm: "Birds of a feather flock together"
  - For a new point, find K nearest training examples
  - Classify based on majority vote (or average for regression)

Key hyperparameter:
  - K: Number of neighbors to consider
    - K=1: Very flexible, likely to overfit
    - K=n: Very rigid, likely to underfit
    - Typically use K = √n (where n = training size) or cross-validate

Advantages:
  - Simple to understand and implement
  - No training phase (lazy learner)
  - Works well for non-linear decision boundaries

Disadvantages:
  - Slow predictions (computes distance to all training points)
  - Requires scaling (distance-based!)
  - High memory usage (stores all training data)
  - Struggles in high dimensions (curse of dimensionality)

Why scaling is critical for KNN:
  - Feature with range [0, 1000] dominates feature with range [0, 1]
  - Scaling ensures all features contribute equally to distance
"""

print("\n" + "=" * 60)
print("K-NEAREST NEIGHBORS")
print("=" * 60)

# Must use SCALED data for KNN!
scaler_knn = StandardScaler()
X_train_knn = scaler_knn.fit_transform(X_train_tree)
X_test_knn = scaler_knn.transform(X_test_tree)

# Train with different K values
knn_k3 = KNeighborsClassifier(n_neighbors=3)
knn_k5 = KNeighborsClassifier(n_neighbors=5)
knn_k15 = KNeighborsClassifier(n_neighbors=15)

knn_k3.fit(X_train_knn, y_train_tree)
knn_k5.fit(X_train_knn, y_train_tree)
knn_k15.fit(X_train_knn, y_train_tree)

# Compare different K values
print(f"K=3 (flexible):")
print(f"  Train accuracy: {knn_k3.score(X_train_knn, y_train_tree):.4f}")
print(f"  Test accuracy: {knn_k3.score(X_test_knn, y_test_tree):.4f}")

print(f"\nK=5 (balanced):")
print(f"  Train accuracy: {knn_k5.score(X_train_knn, y_train_tree):.4f}")
print(f"  Test accuracy: {knn_k5.score(X_test_knn, y_test_tree):.4f}")

print(f"\nK=15 (rigid):")
print(f"  Train accuracy: {knn_k15.score(X_train_knn, y_train_tree):.4f}")
print(f"  Test accuracy: {knn_k15.score(X_test_knn, y_test_tree):.4f}")

# Effect of scaling
print("\nEffect of scaling:")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train_tree, y_train_tree)

print(f"KNN without scaling: {knn_unscaled.score(X_test_tree, y_test_tree):.4f}")
print(f"KNN with scaling: {knn_k5.score(X_test_knn, y_test_tree):.4f}")


# SECTION 3.5: RANDOM FOREST
# ===========================
"""
Use for: Classification or regression
Algorithm: Ensemble of decision trees
  - Train multiple trees on random subsets of data
  - Each tree votes (classification) or averages (regression)
  - Aggregate votes/predictions for final result

Key hyperparameters:
  - n_estimators: Number of trees (more = usually better, but slower)
  - max_depth: Maximum depth per tree (controls complexity)
  - min_samples_split: Minimum samples to split
  - max_features: Number of features per split ('sqrt', 'log2', or int)

Advantages:
  - Reduces overfitting compared to single tree
  - Handles non-linear relationships
  - Feature importance from all trees
  - Robust and generally good performance
  - No scaling needed
  - Parallelizable (multiple trees trained independently)

Disadvantages:
  - Less interpretable than single tree
  - Slower predictions than single tree
  - High memory usage
  - Can still overfit if max_depth too high

Why ensemble works:
  - Individual trees make different errors
  - Averaging reduces variance without increasing bias
  - "Wisdom of crowds" principle
"""

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

# Create Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,  # 100 trees
    max_depth=5,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

rf_model.fit(X_train_tree, y_train_tree)

# Predictions
y_pred_rf = rf_model.predict(X_test_tree)

# Evaluate
accuracy_rf = accuracy_score(y_test_tree, y_pred_rf)
feature_importance_rf = rf_model.feature_importances_

print(f"Accuracy: {accuracy_rf:.4f}")
print(f"\nFeature Importances (from 100 trees):")
for name, importance in zip(iris_data.feature_names, feature_importance_rf):
    print(f"  {name}: {importance:.4f}")

# Compare ensemble vs single tree
print(f"\nEnsemble effect:")
print(f"Single tree accuracy: {accuracy_tree:.4f}")
print(f"Random forest accuracy: {accuracy_rf:.4f}")
print("(Forest often better due to reduced variance)")

# Effect of number of trees
print(f"\nEffect of n_estimators:")
for n_trees in [10, 50, 100, 200]:
    rf_temp = RandomForestClassifier(n_estimators=n_trees, max_depth=5, random_state=42)
    rf_temp.fit(X_train_tree, y_train_tree)
    print(f"  {n_trees} trees: {rf_temp.score(X_test_tree, y_test_tree):.4f}")


# ============================================================================
# SUMMARY & QUICK REFERENCE
# ============================================================================

"""
ALGORITHM CHEATSHEET
====================

Linear Regression:
  ├─ Use: Continuous predictions
  ├─ Needs scaling: Yes (for regularized variants)
  ├─ Interpretable: Yes (coefficients show feature effect)
  └─ Key metrics: MSE, RMSE, R²

Logistic Regression:
  ├─ Use: Binary/multiclass classification
  ├─ Needs scaling: Yes (important for convergence)
  ├─ Interpretable: Yes
  └─ Key metrics: Accuracy, Precision, Recall, F1, AUC

Decision Tree:
  ├─ Use: Classification/regression, interpretability priority
  ├─ Needs scaling: No
  ├─ Interpretable: Yes (visual tree structure)
  └─ Key challenge: Overfitting (control with max_depth)

KNN:
  ├─ Use: Non-linear patterns, small-medium datasets
  ├─ Needs scaling: YES (critical!)
  ├─ Interpretable: Somewhat (show K neighbors)
  └─ Key challenge: Choosing K, slow predictions

Random Forest:
  ├─ Use: General purpose, good default choice
  ├─ Needs scaling: No
  ├─ Interpretable: Moderate (feature importance)
  └─ Key advantage: Reduces overfitting of single trees


KEY WORKFLOWS
=============

1. Data Preparation Pipeline:
   Split Data → Fit Scaler/Imputer on Train → Transform Train & Test

2. Model Evaluation:
   Train/Test Split → Train Model → Predictions → Calculate Metrics

3. Preventing Data Leakage:
   - Split FIRST, before any preprocessing
   - Fit scalers/imputers on TRAIN data only
   - Transform both TRAIN and TEST using SAME fitted transformer

4. Model Selection:
   - Start simple (Linear Regression, Logistic Regression)
   - Try tree-based if non-linear patterns exist
   - Use ensemble (Random Forest) for robust performance
   - Tune hyperparameters on validation set


IMPORTANT REMINDERS
===================

1. Trailing underscore (_):
   - No underscore: hyperparameters (user sets)
   - With underscore: learned attributes (fit() creates)

2. Always use same transformer on test:
   scaler.fit(X_train)
   X_train = scaler.transform(X_train)  # OK
   X_test = scaler.transform(X_test)    # Use SAME scaler!
   # NOT: scaler.fit(X_test); scaler.transform(X_test)

3. Why stratified split:
   train_test_split(..., stratify=y)
   Ensures class distribution same in train/test

4. Why fit only on training:
   - Tests generalization to unseen data
   - Prevents data leakage
   - Gives realistic performance estimate
"""

print("\n" + "=" * 60)
print("Scikit-learn revision materials complete!")
print("=" * 60)
