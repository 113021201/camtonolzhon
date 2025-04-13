
# ===============================
# MIDTERM PROJECT - AI Ghibli Trend Analysis
# Classification: Predicting Popular Images
# Student: [Your Name]
# Course: Data Visualization
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Step 1: Load and Inspect Data
# -------------------------------

df = pd.read_csv("ai_ghibli_trend_dataset_v2.csv")
print("Initial shape:", df.shape)
print(df.head())

# -------------------------------
# Step 2: Create Target Variable
# -------------------------------

df['is_popular'] = (df['likes'] > 1000).astype(int)
df[['res_width', 'res_height']] = df['resolution'].str.split('x', expand=True).astype(int)
df['is_hand_edited'] = df['is_hand_edited'].map({'Yes': 1, 'No': 0})
df['ethical_concerns_flag'] = df['ethical_concerns_flag'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, columns=['platform'], drop_first=True)

# Drop irrelevant or high-cardinality columns
df_model = df.drop(columns=[
    'image_id', 'user_id', 'prompt', 'resolution',
    'creation_date', 'top_comment', 'likes'
])

X = df_model.drop(columns=['is_popular'])
y = df_model['is_popular']

# -------------------------------
# Step 3: Preprocessing
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Step 4: Model Training
# -------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

# -------------------------------
# Step 5: Visualization
# -------------------------------

# Model performance comparison
summary_data = []
for name, result in results.items():
    report = result["report"]
    summary_data.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Precision (Popular)": report["1"]["precision"],
        "Recall (Popular)": report["1"]["recall"],
        "F1-score (Popular)": report["1"]["f1-score"]
    })
summary_df = pd.DataFrame(summary_data)

# Barplot
plt.figure(figsize=(10, 6))
summary_melted = summary_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=summary_melted, x="Model", y="Score", hue="Metric")
plt.title("Model Performance Comparison (Class: Popular)")
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
for i, (name, result) in enumerate(results.items()):
    cm = result["conf_matrix"]
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=axes[i])
    axes[i].set_title(f"{name} - Confusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------------
# Step 6: Data Exploration
# -------------------------------

# Distribution of engagement
plt.figure(figsize=(10, 4))
sns.histplot(df['likes'], bins=30, kde=True)
plt.title("Distribution of Likes")
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x='is_popular', y='shares', data=df)
plt.title("Shares vs Popularity")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_model.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
