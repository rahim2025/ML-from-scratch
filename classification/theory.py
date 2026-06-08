"""
ML Classification Theory - Core Concepts & Metrics
====================================================

Classification is the task of predicting which category/class an observation belongs to.
This module documents the fundamental concepts, metrics, and decision-making frameworks.
"""

# ============================================================================
# 1. THE DECISION PIPELINE
# ============================================================================
"""
Models do not natively output "Class A" or "Class B". They output probabilities.

The Workflow:
    Features → Model → Probability (0.0 to 1.0) → Threshold → Final Class Prediction

Example:
    - Feature vector [age=35, income=50000] goes through the model
    - Model outputs: P(Fraud) = 0.62
    - Default threshold = 0.5
    - Decision: 0.62 >= 0.5 → Predict "Fraud"

The Default Threshold:
    - Typically 0.5 is used for binary classification
    - But this is NOT always optimal—threshold tuning is crucial
"""

# ============================================================================
# 2. THE CONFUSION MATRIX
# ============================================================================
"""
The foundation for evaluating any classification system.

                    | Predicted Positive | Predicted Negative
    ================================================================
    Actual Positive | TP (True Positive) | FN (False Negative)
                    | Correctly caught   | Type II Error
                    |                    | (Missed case)
    ================================================================
    Actual Negative | FP (False Positive)| TN (True Negative)
                    | Type I Error       | Correctly rejected
                    | (False alarm)      |
    ================================================================

Key Terms:
    - True Positive (TP):     Correctly predicted positive case
    - False Negative (FN):    Missed positive case (Type II Error)
    - False Positive (FP):    Incorrectly flagged negative as positive (Type I Error)
    - True Negative (TN):     Correctly predicted negative case
"""

# ============================================================================
# 3. CORE EVALUATION METRICS
# ============================================================================
"""
These metrics help you understand how well your model performs.
"""

# -------- PRECISION --------
"""
Precision (Quality of Alarms / Positive Predictive Value)

Question: "When the model predicts Positive, how often is it actually correct?"

Formula:
    Precision = TP / (TP + FP)

Mindset: Avoid False Alarms (minimize false positives)

Interpretation:
    - Precision = 0.9 means "90% of the time when the model predicts positive, it's correct"
    - Low precision = Many false alarms

Best For:
    - Spam filtering (deleting a good email = bad user experience)
    - Notification delivery (alerting user for non-critical events = annoying)
    - Legal conviction systems (falsely imprisoning someone = catastrophic)
    
When to Use: When False Positives are costlier than False Negatives
"""

# -------- RECALL --------
"""
Recall (Coverage of Reality / Sensitivity / True Positive Rate)

Question: "Out of all actual Positives, how many did the model catch?"

Formula:
    Recall = TP / (TP + FN)

Mindset: Avoid Missing Cases (minimize false negatives)

Interpretation:
    - Recall = 0.95 means "The model caught 95% of all actual positive cases"
    - Low recall = Many missed cases

Best For:
    - Fraud detection (missing fraud = financial loss)
    - Cancer screening (missing cancer = life-threatening delay)
    - Security threat detection (missing threats = system compromise)

When to Use: When False Negatives are costlier than False Positives
"""

# -------- F1-SCORE --------
"""
F1-Score (The Balanced Metric / Harmonic Mean)

Definition: Harmonic mean of Precision and Recall
    Punishes extreme imbalances between the two metrics

Formula:
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

Interpretation:
    - F1 ranges from 0 (worst) to 1 (perfect)
    - High F1 requires BOTH high precision AND high recall
    - Use when you want a single metric that balances both concerns
    
Example:
    - Precision = 0.8, Recall = 0.6
    - F1 = 2 * (0.8 * 0.6) / (0.8 + 0.6) = 2 * 0.48 / 1.4 ≈ 0.686

When to Use: When you care about both precision and recall equally
             (imbalanced datasets, or when false positives AND false negatives matter)
"""

# ============================================================================
# 4. THE GOLDEN RULE OF TRADE-OFFS
# ============================================================================
"""
Precision and Recall are almost always in opposition.
To decide which to prioritize, ask:

    ❓ "Is it worse to miss a case (False Negative), 
       or to wrongly flag one (False Positive)?"

Decision Framework:

    IF missed case is worse → Prioritize RECALL (Lower threshold)
        Examples: Cancer detection, fraud detection, security threats
        Rationale: False negatives = catastrophic consequences
        Action: Lower decision threshold to catch more positives
    
    IF false alarm is worse → Prioritize PRECISION (Higher threshold)
        Examples: Content moderation, email filtering, legal prosecution
        Rationale: False positives = wasted resources, user frustration
        Action: Raise decision threshold to be more conservative

Cost Considerations:
    - Medical diagnosis: Missing cancer might cost a life (Recall > Precision)
    - Spam filter: Filtering legitimate email annoys users (Precision > Recall)
    - Credit approval: False positive (rejecting good customer) = lost revenue
                       False negative (approving bad customer) = bad debt
                       Usually: need balance or slight Precision bias
"""

# ============================================================================
# 5. THRESHOLD TUNING: THE CONTROL KNOB
# ============================================================================
"""
Critical Insight: You can change model behavior WITHOUT retraining
by simply adjusting the decision threshold.

Think of the threshold as a "Strictness Level":
    - High threshold = Strict mode (only predict positive if very confident)
    - Low threshold = Sensitive mode (predict positive easily)

This is THE KEY to optimizing for your specific use case.
"""

# -------- INCREASING THE THRESHOLD --------
"""
Action: Increase threshold (e.g., 0.5 → 0.8)

What Happens:
    - Model becomes highly strict
    - Only predicts "Positive" if extremely confident
    - Result: Fewer predictions of "Positive"

Metric Changes:
    - Precision ↑ (fewer false alarms)
    - Recall ↓ (more actual cases slip through as false negatives)
    - FP ↓ (fewer false positives)
    - FN ↑ (more false negatives)

Scenario: 
    Original: threshold=0.5 → Precision=0.85, Recall=0.75
    Adjusted: threshold=0.8 → Precision=0.95, Recall=0.60
    
Use Case: When you can't afford false alarms (precision-critical)
"""

# -------- DECREASING THE THRESHOLD --------
"""
Action: Decrease threshold (e.g., 0.5 → 0.2)

What Happens:
    - Model becomes highly sensitive
    - Eagerly triggers "Positive" at slightest hint
    - Result: More predictions of "Positive"

Metric Changes:
    - Recall ↑ (catches almost everything)
    - Precision ↓ (flooded with false alarms)
    - FN ↓ (fewer false negatives)
    - FP ↑ (more false positives)

Scenario:
    Original: threshold=0.5 → Precision=0.85, Recall=0.75
    Adjusted: threshold=0.2 → Precision=0.60, Recall=0.95
    
Use Case: When you can't afford missed cases (recall-critical)
"""

# ============================================================================
# 6. QUICK REFERENCE CHEAT SHEET
# ============================================================================
"""
┌─────────────────────────────────────────────────────────────────┐
│ THRESHOLD TUNING SUMMARY                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ HIGH STRICTNESS (High Threshold):                              │
│   • Precision ↑ | Recall ↓                                     │
│   • Few positives predicted                                    │
│   • Great for: Spam filtering, legal conviction                │
│                                                                 │
│ LOW STRICTNESS (Low Threshold):                                │
│   • Recall ↑ | Precision ↓                                     │
│   • Many positives predicted                                   │
│   • Great for: Cancer detection, fraud detection               │
│                                                                 │
│ USE CASES:                                                      │
│   • Cancer/Fraud/Security: Use LOW threshold (maximize recall)  │
│   • Spam/Marketing/Moderation: Use HIGH threshold               │
│     (maximize precision)                                        │
│   • Balanced scenarios: Use F1-score or domain-specific costs   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

# ============================================================================
# 7. HELPER FUNCTIONS FOR CALCULATIONS
# ============================================================================

def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix components from true labels and predictions.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        dict with TP, FN, FP, TN counts
    """
    tp = sum((y_true[i] == 1) and (y_pred[i] == 1) for i in range(len(y_true)))
    fn = sum((y_true[i] == 1) and (y_pred[i] == 0) for i in range(len(y_true)))
    fp = sum((y_true[i] == 0) and (y_pred[i] == 1) for i in range(len(y_true)))
    tn = sum((y_true[i] == 0) and (y_pred[i] == 0) for i in range(len(y_true)))
    
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn}


def precision(tp, fp):
    """
    Calculate precision: TP / (TP + FP)
    
    Args:
        tp: True positives
        fp: False positives
    
    Returns:
        Precision score (0 to 1), or None if denominator is 0
    """
    if tp + fp == 0:
        return None
    return tp / (tp + fp)


def recall(tp, fn):
    """
    Calculate recall: TP / (TP + FN)
    
    Args:
        tp: True positives
        fn: False negatives
    
    Returns:
        Recall score (0 to 1), or None if denominator is 0
    """
    if tp + fn == 0:
        return None
    return tp / (tp + fn)


def f1_score(precision_val, recall_val):
    """
    Calculate F1-score: 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        precision_val: Precision score
        recall_val: Recall score
    
    Returns:
        F1-score (0 to 1), or None if either input is None or sum is 0
    """
    if precision_val is None or recall_val is None:
        return None
    if precision_val + recall_val == 0:
        return None
    return 2 * (precision_val * recall_val) / (precision_val + recall_val)


def apply_threshold(probabilities, threshold=0.5):
    """
    Convert probability predictions to binary predictions using a threshold.
    
    Args:
        probabilities: Array of probability values (0.0 to 1.0)
        threshold: Decision threshold (default 0.5)
    
    Returns:
        Array of binary predictions (0 or 1)
    """
    return [1 if p >= threshold else 0 for p in probabilities]


def threshold_sweep_analysis(y_true, y_probs, thresholds=None):
    """
    Analyze precision, recall, and F1-score across multiple thresholds.
    
    Args:
        y_true: Array of true binary labels
        y_probs: Array of predicted probabilities (0.0 to 1.0)
        thresholds: List of thresholds to test (default: 0.1 to 0.9)
    
    Returns:
        List of dicts with threshold, precision, recall, f1-score
    """
    if thresholds is None:
        thresholds = [i * 0.1 for i in range(1, 10)]
    
    results = []
    for threshold in thresholds:
        y_pred = apply_threshold(y_probs, threshold)
        cm = compute_confusion_matrix(y_true, y_pred)
        
        p = precision(cm["TP"], cm["FP"])
        r = recall(cm["TP"], cm["FN"])
        f1 = f1_score(p, r)
        
        results.append({
            "threshold": threshold,
            "precision": p,
            "recall": r,
            "f1_score": f1
        })
    
    return results


# ============================================================================
# 8. EXAMPLE: PRACTICAL USAGE
# ============================================================================
"""
Example Scenario: Fraud Detection System
=========================================

Data:
    y_true = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]  # Actual labels (1=fraud, 0=legit)
    y_probs = [0.9, 0.1, 0.8, 0.2, 0.7, 0.15, 0.3, 0.85, 0.05, 0.75]  # Model outputs

Step 1: Apply default threshold (0.5)
    y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    Confusion matrix: TP=5, FN=0, FP=0, TN=5
    Precision = 5/(5+0) = 1.0
    Recall = 5/(5+0) = 1.0
    F1 = 1.0

Step 2: Try higher threshold (0.8) to reduce false alarms
    y_pred = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    Confusion matrix: TP=2, FN=3, FP=0, TN=5
    Precision = 2/(2+0) = 1.0 (no false alarms!)
    Recall = 2/(2+3) = 0.4 (but we missed 3 fraud cases)
    F1 = 0.571

Step 3: Decision - is 0.8 threshold better?
    Question: Is it worse to miss fraud (FN) or raise false alarm (FP)?
    Answer: Missing fraud is worse (financial loss)
    Decision: Keep lower threshold (0.5) or go even lower
    
This is the power of threshold tuning—no retraining needed!
"""

# ============================================================================
# 9. KEY TAKEAWAYS
# ============================================================================
"""
1. Classification produces probabilities, not class labels
2. Confusion matrix is your foundation for understanding model performance
3. Precision and Recall are in opposition (trade-off)
4. F1-score is the harmonic mean for balanced evaluation
5. Golden rule: Ask "what's worse—missed case or false alarm?"
6. Threshold tuning lets you optimize for YOUR specific use case
7. High threshold = High precision (strict, fewer positives)
8. Low threshold = High recall (lenient, more positives)
9. Different domains have different requirements:
    - Medical: Recall >> Precision (can't miss disease)
    - Spam: Precision >> Recall (can't bother users)
    - Balanced: F1-score or cost-weighted optimization
"""
