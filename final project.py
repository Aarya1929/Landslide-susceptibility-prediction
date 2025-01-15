import pandas as pd
import numpy as np
from scipy import stats
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
import os

# Set the number of CPU cores you want to use (e.g., 4 cores)
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load the dataset from a CSV file
df = pd.read_csv(r'df final.csv')

# Drop latitude and longitude columns if they exist
columns_to_drop = ['Latitude', 'Longitude']
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=existing_columns_to_drop) if existing_columns_to_drop else df

# Detect and remove outliers using Z-score
def detect_outliers_zscore(df, threshold=3):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    outliers = (z_scores > threshold).any(axis=1)
    return outliers

outliers = detect_outliers_zscore(df)
df_cleaned = df[~outliers]

# Recompute class distribution after removing outliers
target_column = 'Risk'
class_counts_cleaned = df_cleaned[target_column].value_counts()

X = df_cleaned.drop(target_column, axis=1)
y = df_cleaned[target_column]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Initialize SMOTE for handling class imbalance
smote = SMOTE(random_state=42, k_neighbors=min(5, y.value_counts().min() - 1))
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to calculate TSS
def calculate_tss(conf_matrix):
    TP = conf_matrix[1, 1]  # True Positive
    TN = conf_matrix[0, 0]  # True Negative
    FP = conf_matrix[0, 1]  # False Positive
    FN = conf_matrix[1, 0]  # False Negative
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    tss = sensitivity + specificity - 1
    return tss

# Function to evaluate a model
def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name):
    # Fit the model
    model.fit(X_train_scaled, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)  # Get predicted probabilities

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate AUC-ROC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except ValueError:
        roc_auc = np.nan  # Handle the case where AUC cannot be computed

    tss = calculate_tss(conf_matrix)
    
    print(f"{model_name} Evaluation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"TSS: {tss:.2f}")
    print(f"AUC-ROC: {roc_auc:.2f}")
    
    # Classification report
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print(f"{model_name} Confusion Matrix:")
    print(conf_matrix)

# Logistic Regression
logreg = LogisticRegression(random_state=42, max_iter=1000)
evaluate_model(logreg, X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression")

# Support Vector Machine (SVM)
svm = SVC(random_state=42, probability=True)  # Enable probability estimates
evaluate_model(svm, X_train_scaled, y_train, X_test_scaled, y_test, "Support Vector Machine")

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn, X_train_scaled, y_train, X_test_scaled, y_test, "K-Nearest Neighbors")

# Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=100)
evaluate_model(rf, X_train_scaled, y_train, X_test_scaled, y_test, "Random Forest")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
models = ['RF', 'SVM', 'KNN', 'LR']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'TSS']
values = [
    [96.85, 0.97, 0.97, 0.97, 1.00],
    [93.30, 0.94, 0.93, 0.93, 0.99],
    [94.94, 0.95, 0.95, 0.95, 0.96],
    [94.25, 0.94, 0.94, 0.94, 0.99]
]

# Convert to DataFrame
df = pd.DataFrame(values, columns=metrics, index=models)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
df.plot(kind='bar', ax=ax, width=0.7)   # Adjusted bar width
plt.title('Model Performance Metrics', fontsize=10)
plt.xlabel('Models', fontsize=9)
plt.ylabel('Values', fontsize=9)
plt.ylim(0, 1.2)  # Increased range slightly
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)  # Optional: Line at 1
plt.legend(title='Metrics', fontsize=8)
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.tight_layout()  # Ensures everything fits in a smaller plot
plt.show()
