#!/usr/bin/env python3
"""
Generate ML classification charts for shadow economy detection using synthetic transaction data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create charts directory
charts_dir = "/root/hub/research/shadow-economy-dynamics/charts"
os.makedirs(charts_dir, exist_ok=True)

# Generate synthetic transaction data
np.random.seed(42)
n_samples = 10000
n_features = 10

# Features: transaction amount, frequency, time, location entropy, etc.
X = np.random.randn(n_samples, n_features)
# Add some informative features
X[:, 0] = np.random.exponential(scale=2.0, size=n_samples)  # transaction amount (heavy-tailed)
X[:, 1] = np.random.lognormal(mean=0, sigma=1, size=n_samples)  # frequency
# Create binary target: suspicious (1) vs normal (0)
# True labels depend on a linear combination of features plus noise
coef = np.random.randn(n_features)
logits = X.dot(coef) + np.random.randn(n_samples) * 0.5
y = (logits > 0).astype(int)
# Make imbalanced (5% suspicious)
y = (logits > np.percentile(logits, 95)).astype(int)
print(f"Dataset shape: {X.shape}, suspicious ratio: {y.mean():.3f}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

results = {}
print("Training models...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    results[name] = y_pred_proba
    print(f"{name} trained.")

# 1. ROC Curve
plt.figure(figsize=(8, 6))
for name, y_pred_proba in results.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: ML Models for Suspicious Transaction Detection')
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(charts_dir, 'roc_curve_transaction_models.png')
plt.savefig(roc_path, dpi=300)
print(f"Saved ROC curve to {roc_path}")
plt.close()

# 2. Feature Importance (Random Forest)
rf = models["Random Forest"]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = [f'Feature {i+1}' for i in range(n_features)]

plt.figure(figsize=(8, 6))
plt.bar(range(n_features), importances[indices], align='center')
plt.xticks(range(n_features), [features[i] for i in indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance (Synthetic Transaction Data)')
plt.tight_layout()
feat_path = os.path.join(charts_dir, 'feature_importance_rf_transaction.png')
plt.savefig(feat_path, dpi=300)
print(f"Saved feature importance to {feat_path}")
plt.close()

# 3. Precision-Recall Curve
plt.figure(figsize=(8, 6))
for name, y_pred_proba in results.items():
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    plt.plot(recall, precision, lw=2, label=f'{name} (AP = {avg_precision:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves (Imbalanced Data)')
plt.legend(loc="upper right")
plt.tight_layout()
pr_path = os.path.join(charts_dir, 'precision_recall_transaction.png')
plt.savefig(pr_path, dpi=300)
print(f"Saved Precision-Recall curve to {pr_path}")
plt.close()

# 4. Confusion Matrix for best model (XGBoost)
best_model = models["XGBoost"]
y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Suspicious'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - XGBoost')
cm_path = os.path.join(charts_dir, 'confusion_matrix_xgboost_transaction.png')
plt.savefig(cm_path, dpi=300)
print(f"Saved confusion matrix to {cm_path}")
plt.close()

# 5. Distribution of transaction amounts (feature 0)
plt.figure(figsize=(8, 6))
plt.hist(X[:, 0], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
plt.xlabel('Transaction Amount (scaled)')
plt.ylabel('Frequency')
plt.title('Distribution of Synthetic Transaction Amounts')
plt.tight_layout()
dist_path = os.path.join(charts_dir, 'transaction_amount_distribution.png')
plt.savefig(dist_path, dpi=300)
print(f"Saved distribution to {dist_path}")
plt.close()

print("All charts generated.")
