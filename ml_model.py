import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
)
import seaborn as sns

# Load the processed Titanic dataset
df = pd.read_csv("titanic_numerology.csv")

# Features: Only use numerology-based columns
X = df[['Name_Numerology', 'Soul_Number', 'Personality_Number', 'Name_Length']]
y = df['Survived']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Models
metrics = {
    "Logistic Regression": {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr),
        "recall": recall_score(y_test, y_pred_lr),
        "conf_matrix": confusion_matrix(y_test, y_pred_lr),
    },
    "Random Forest": {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "conf_matrix": confusion_matrix(y_test, y_pred_rf),
    }
}

# Print Metrics
for model_name, scores in metrics.items():
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {scores['accuracy']:.2f}")
    print(f"Precision: {scores['precision']:.2f}")
    print(f"Recall: {scores['recall']:.2f}")

# Plot Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(metrics['Logistic Regression']['conf_matrix'], annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title("Logistic Regression Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(metrics['Random Forest']['conf_matrix'], annot=True, fmt='d', ax=axes[1], cmap='Greens')
axes[1].set_title("Random Forest Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("ml_confusion_matrices.png")
plt.show()
