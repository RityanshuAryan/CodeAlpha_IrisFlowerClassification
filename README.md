# CodeAlpha_IrisFlowerClassification

**Data Science Internship — CodeAlpha**

Machine learning project that classifies Iris flowers (*setosa*, *versicolor*, *virginica*)
into their correct species based on sepal and petal measurements.

## 📊 Dataset

The classic Iris dataset (150 samples, 4 numeric features, 3 balanced classes), loaded via
`sklearn.datasets.load_iris`. A CSV copy is included as `iris_dataset.csv`.

| Feature | Description |
|---|---|
| sepal length (cm) | Length of the sepal |
| sepal width (cm) | Width of the sepal |
| petal length (cm) | Length of the petal |
| petal width (cm) | Width of the petal |
| species | Target label (setosa / versicolor / virginica) |

## 🔍 Approach

1. **EDA** — class balance check, summary statistics, pairplot, correlation heatmap, and
   boxplots by species to understand which features separate the classes best.
2. **Preprocessing** — 80/20 stratified train/test split, feature standardization.
3. **Modeling** — trained and compared 5 classifiers with 5-fold cross-validation:
   Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and SVM (RBF kernel).
4. **Evaluation** — accuracy, classification report (precision/recall/F1), and a confusion matrix
   for the best model.

## ✅ Results

| Model | Test Accuracy | CV Mean Accuracy |
|---|---|---|
| **SVM (RBF kernel)** | **96.7%** | 96.7% |
| Logistic Regression | 93.3% | 96.0% |
| K-Nearest Neighbors | 93.3% | 96.0% |
| Decision Tree | 93.3% | 95.3% |
| Random Forest | 90.0% | 96.7% |

The SVM with an RBF kernel generalized best. Its only errors were between *versicolor* and
*virginica* — the one genuinely overlapping boundary in this dataset (see `confusion_matrix.png`).

## 📁 Files

```
iris_classification.py        # full pipeline: EDA → preprocessing → training → evaluation
iris_dataset.csv              # dataset
pairplot.png                  # feature pairplot by species
correlation_heatmap.png       # feature correlation matrix
boxplots.png                  # feature distributions by species
model_accuracy_comparison.png # bar chart comparing all 5 models
confusion_matrix.png          # confusion matrix for the best model
model_comparison.csv          # accuracy/CV scores table
results_summary.txt           # full text summary of results
best_iris_model.joblib        # trained SVM model
scaler.joblib                 # fitted StandardScaler
```

## ▶️ Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python3 iris_classification.py
```

## 🛠 Tools

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn

---
*Task 1 of the CodeAlpha Data Science Internship.*
