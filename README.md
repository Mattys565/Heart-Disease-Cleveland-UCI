# Heart Disease Prediction — Cleveland UCI

End-to-end ML pipeline on the Cleveland Heart Disease dataset: EDA, baseline models, hyperparameter tuning, feature importance, and SHAP analysis.

---

## Dataset

Cleveland subset of the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) — 303 patients, 13 clinical features, binary target (0 = no disease, 1 = disease).

Missing values (encoded as `?`) are dropped. The other subsets (Hungary, VA, Switzerland) were unusable due to ~90% missing data on the most important features.

Place the data file at `data/processed.cleveland.data`.

---

## What's in the notebook

1. **EDA** — age/sex distributions by target, correlation heatmap
2. **Preprocessing** — stratified train/test split, StandardScaler fit on train only
3. **Baseline models** — XGBoost, KNN, Random Forest, Logistic Regression, SVC, Decision Tree with 5-fold stratified CV
4. **Hyperparameter tuning** — RandomizedSearchCV on SVC, LR, XGBoost, optimizing for recall
5. **Feature importance** — feature_importances_, coefficients, permutation importance
6. **SHAP** — bar plot, summary plot, waterfall plots, dependence plot

---

## Main findings

`ca`, `thal` and `cp` are the strongest predictors across every method tested. The three tuned models (SVC, LR, XGBoost) all land at similar performance, which makes sense — the signal in this dataset is mostly linear.

Recall was chosen as the optimization metric: missing a sick patient is a bigger problem than flagging a healthy one.

`fbs`, `chol`, `restecg` and `trestbps` show near-zero importance across the board — could probably be dropped.

---

## Stack

```
pandas / numpy
matplotlib / seaborn
scikit-learn
xgboost
shap
```

---

## Run it

```bash
git clone https://github.com/Mattys565/heart-disease-cleveland.git
cd heart-disease-cleveland
pip install -r requirements.txt
jupyter notebook Heart_disease.ipynb
```
