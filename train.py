import pandas as pd

# Load the dataset
df = pd.read_csv("data/diabetes.csv")

# First 5 rows
print("First 5 Rows")
print(df.head())

print("\n------------------------")

# Shape of dataset
print("Shape of Dataset")
print(df.shape)

print("\n------------------------")

# Column names
print("Columns")
print(df.columns)

print("\n------------------------")

# Information about dataset
print("Dataset Information")
print(df.info())

print("\n------------------------")

# Statistical summary
print("Statistical Summary")
print(df.describe())
print("\n------------------------")

# Check missing values
print("Missing Values")
print(df.isnull().sum())

print("\n------------------------")

# Count diabetic and non-diabetic patients
print("Outcome Counts")
print(df["Outcome"].value_counts())

print("\n------------------------")

# Percentage distribution
print("Outcome Percentage")
print(df["Outcome"].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of all featuresprint("Before Histogram")

df.hist(figsize=(12,8), bins=20)
plt.suptitle("Feature Distributions")
plt.tight_layout()

print("Showing Histogram")
plt.show()

print("Histogram Closed")

print("Before Heatmap")

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

print("Showing Heatmap")
plt.title("Correlation Heatmap")
plt.show()

print("Heatmap Closed")
import numpy as np

# Columns where 0 is not medically valid
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace 0 with NaN
df[zero_columns] = df[zero_columns].replace(0, np.nan)

print("\nMissing Values After Replacing 0:")
print(df.isnull().sum())
# ==========================================
# Fill Missing Values
# ==========================================

df.fillna(df.median(numeric_only=True), inplace=True)

print("\nMissing Values After Filling:")
print(df.isnull().sum())

# ==========================================
# Features (X) and Target (y)
# ==========================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFirst 5 rows of Features")
print(X.head())

print("\nFirst 5 Target Values")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)
from sklearn.preprocessing import StandardScaler

# Create scaler object
scaler = StandardScaler()

# Fit only on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

print("\nTraining Data After Scaling:")
print(X_train_scaled[:5])

print("\nTesting Data After Scaling:")
print(X_test_scaled[:5])
# ==========================================
# Train Logistic Regression Model
# ==========================================

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression Model Trained Successfully!")

# ==========================================
# Make Predictions
# ==========================================

y_pred = model.predict(X_test_scaled)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("\nActual Values:")
print(y_test.values[:10])
# ==========================================
# Model Accuracy
# ==========================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

from sklearn.metrics import classification_report

print("\nClassification Report")
print(classification_report(y_test, y_pred))
# ==========================================
# Random Forest Classifier
# ==========================================

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:", rf_accuracy)

# Classification Report
from sklearn.metrics import classification_report

print("\nRandom Forest Classification Report")
print(classification_report(y_test, rf_predictions))

# Confusion Matrix
from sklearn.metrics import confusion_matrix

print("\nRandom Forest Confusion Matrix")
print(confusion_matrix(y_test, rf_predictions))
# ==========================================
# Save Model
# ==========================================

import joblib

joblib.dump(model, "models/diabetes_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel Saved Successfully!")