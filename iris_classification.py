"""
TASK 1: Iris Flower Classification
CodeAlpha Data Science Internship

Pipeline: load data -> EDA -> preprocess -> train multiple models ->
evaluate -> pick best -> save artifacts.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

OUT = "/home/claude/project/task1_iris"
sns.set_style("whitegrid")

# ---------- 1. Load data ----------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.to_csv(f"{OUT}/iris_dataset.csv", index=False)
print("Dataset shape:", df.shape)
print(df.head())

# ---------- 2. EDA ----------
print("\nClass distribution:\n", df["species"].value_counts())
print("\nSummary stats:\n", df.describe())

fig = sns.pairplot(df, hue="species", diag_kind="hist", palette="Set2")
fig.fig.suptitle("Iris Feature Pairplot by Species", y=1.02)
fig.savefig(f"{OUT}/pairplot.png", dpi=150, bbox_inches="tight")
plt.close("all")

plt.figure(figsize=(7, 5))
sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df.melt(id_vars="species"), x="variable", y="value", hue="species")
plt.title("Feature Distributions by Species")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{OUT}/boxplots.png", dpi=150)
plt.close()

# ---------- 3. Preprocess ----------
X = df.drop(columns="species")
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- 4. Train & compare multiple models ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)
    results.append({"Model": name, "Test Accuracy": acc, "CV Mean Accuracy": cv_scores.mean()})

results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
print("\nModel comparison:\n", results_df)

# ---------- 5. Best model deep-dive ----------
best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]
preds = best_model.predict(X_test_s)

print(f"\nBest model: {best_name}")
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification report:\n", classification_report(y_test, preds))

cm = confusion_matrix(y_test, preds, labels=iris.target_names)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.close()

# Model comparison bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Test Accuracy", y="Model", palette="viridis")
plt.xlim(0.8, 1.01)
plt.title("Model Accuracy Comparison — Iris Classification")
plt.tight_layout()
plt.savefig(f"{OUT}/model_accuracy_comparison.png", dpi=150)
plt.close()

joblib.dump(best_model, f"{OUT}/best_iris_model.joblib")
joblib.dump(scaler, f"{OUT}/scaler.joblib")

with open(f"{OUT}/results_summary.txt", "w") as f:
    f.write("TASK 1: IRIS FLOWER CLASSIFICATION - RESULTS SUMMARY\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Dataset: {df.shape[0]} samples, {df.shape[1]-1} features, 3 classes\n\n")
    f.write("Model comparison:\n")
    f.write(results_df.to_string(index=False) + "\n\n")
    f.write(f"Best model: {best_name}\n")
    f.write(f"Test accuracy: {accuracy_score(y_test, preds):.4f}\n\n")
    f.write("Classification report:\n")
    f.write(classification_report(y_test, preds))

print("\nTask 1 complete. Artifacts saved to", OUT)
