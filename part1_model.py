import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

# Load dataset
df = pd.read_csv("orders_dataset.csv")

# Features and target
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

# Column types
numeric_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given"
]

categorical_features = [
    "product_category",
    "payment_method"
]

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Baseline
baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DummyClassifier(strategy="most_frequent"))
])

baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("BASELINE")
print("Accuracy:", accuracy_score(y_test, baseline_pred))
print("F1:", f1_score(y_test, baseline_pred, zero_division=0))

# Logistic Regression
logistic = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    ))
])

logistic.fit(X_train, y_train)

prob = logistic.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, pred))
print("F1:", f1_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
# Threshold sweep
print("\nTHRESHOLD SWEEP")

best_threshold = None
best_f1 = -1

for threshold in [i / 100 for i in range(10, 91, 2)]:
    threshold_pred = (prob >= threshold).astype(int)

    f1 = f1_score(y_test, threshold_pred, zero_division=0)
    recall = recall_score(y_test, threshold_pred, zero_division=0)
    precision = precision_score(y_test, threshold_pred, zero_division=0)

    print(
        f"Threshold={threshold:.2f} "
        f"F1={f1:.4f} "
        f"Recall={recall:.4f} "
        f"Precision={precision:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest threshold:", best_threshold)
print("Best F1:", best_f1)
# ============================================================
# TASK 6 - RANDOM FOREST + GRID SEARCH
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

# Parameters required by the assignment
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [6, 10, None]
}

# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Grid Search using ROC-AUC
grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

# Train GridSearchCV on training data only
grid_search.fit(X_train, y_train)

# Best parameters
print("\nRANDOM FOREST")
print("Best parameters:", grid_search.best_params_)

# Best cross-validation ROC-AUC
print("Best CV ROC-AUC:", grid_search.best_score_)

# Evaluate winning model on held-out test set
best_rf = grid_search.best_estimator_

rf_prob = best_rf.predict_proba(X_test)[:, 1]

test_roc_auc = roc_auc_score(y_test, rf_prob)

print("Test ROC-AUC:", test_roc_auc)
# ============================================================
# TASK 7 - FEATURE IMPORTANCE + PERMUTATION IMPORTANCE
# ============================================================

from sklearn.inspection import permutation_importance

# Get feature names after preprocessing
feature_names = best_rf.named_steps["preprocess"].get_feature_names_out()

# Get Random Forest model
rf_model = best_rf.named_steps["model"]

# Impurity-based feature importance
importances = rf_model.feature_importances_

# Sort features by importance
sorted_idx = importances.argsort()[::-1]

print("\nTOP 5 FEATURE IMPORTANCES")

top5_features = []

for i in sorted_idx[:5]:
    feature = feature_names[i]
    importance = importances[i]
    top5_features.append(feature)
    print(f"{feature}: {importance:.6f}")
# ============================================================
# TASK 7 - FEATURE IMPORTANCE + PERMUTATION IMPORTANCE
# ============================================================

from sklearn.inspection import permutation_importance

# Get feature names after preprocessing
feature_names = best_rf.named_steps["preprocess"].get_feature_names_out()

# Get Random Forest model
rf_model = best_rf.named_steps["model"]

# ------------------------------------------------------------
# 1. IMPURITY-BASED FEATURE IMPORTANCE
# ------------------------------------------------------------

importances = rf_model.feature_importances_

sorted_idx = importances.argsort()[::-1]

print("\nTOP 5 FEATURE IMPORTANCES")

top5_features = []

for i in sorted_idx[:5]:
    feature = feature_names[i]
    importance = importances[i]

    top5_features.append(feature)

    print(f"{feature}: {importance:.6f}")


# ------------------------------------------------------------
# 2. PERMUTATION IMPORTANCE ON HELD-OUT TEST SET
# ------------------------------------------------------------

print("\nPERMUTATION IMPORTANCE - TOP 5")

perm = permutation_importance(
    best_rf,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

raw_features = list(X_test.columns)

perm_importance = perm.importances_mean

perm_sorted_idx = perm_importance.argsort()[::-1]

for i in perm_sorted_idx[:5]:
    print(
        f"{raw_features[i]}: "
        f"{perm_importance[i]:.6f}"
    )


# ------------------------------------------------------------
# 3. AGGREGATE IMPURITY IMPORTANCE TO ORIGINAL FEATURES
# ------------------------------------------------------------

aggregated_importance = {}

for name, value in zip(feature_names, importances):

    # Remove preprocessing prefix such as num__ or cat__
    original_name = name.split("__", 1)[-1]

    # Match one-hot encoded categorical features
    matched_feature = None

    for raw_name in raw_features:

        if (
            original_name == raw_name
            or original_name.startswith(raw_name + "_")
        ):
            matched_feature = raw_name
            break

    if matched_feature is None:
        matched_feature = original_name

    aggregated_importance[matched_feature] = (
        aggregated_importance.get(matched_feature, 0)
        + value
    )


# ------------------------------------------------------------
# 4. TOP 5 ORIGINAL FEATURES
# ------------------------------------------------------------

top5_raw = sorted(
    aggregated_importance,
    key=aggregated_importance.get,
    reverse=True
)[:5]


# ------------------------------------------------------------
# 5. COMPARE IMPURITY VS PERMUTATION
# ------------------------------------------------------------

print("\nTOP 5 COMPARISON")

for feature in top5_raw:

    if feature in raw_features:

        idx = raw_features.index(feature)

        print(
            f"{feature} | "
            f"Impurity={aggregated_importance[feature]:.6f} | "
            f"Permutation={perm_importance[idx]:.6f}"
        )
        # ============================================================
# TASK 8 - SUBGROUP / ROOT-CAUSE ANALYSIS
# ============================================================

from sklearn.metrics import recall_score, precision_score

# Predictions from the winning Random Forest
y_test_pred = best_rf.predict(X_test)

# Overall performance
overall_recall = recall_score(y_test, y_test_pred, zero_division=0)
overall_precision = precision_score(y_test, y_test_pred, zero_division=0)

print("\nOVERALL TEST PERFORMANCE")
print(f"Recall: {overall_recall:.4f}")
print(f"Precision: {overall_precision:.4f}")


# ------------------------------------------------------------
# Performance by PRODUCT CATEGORY
# ------------------------------------------------------------

print("\nPERFORMANCE BY PRODUCT_CATEGORY")

product_results = []

for category in X_test["product_category"].dropna().unique():

    mask = X_test["product_category"] == category

    y_true_group = y_test.loc[mask]
    y_pred_group = y_test_pred[mask]

    recall = recall_score(
        y_true_group,
        y_pred_group,
        zero_division=0
    )

    precision = precision_score(
        y_true_group,
        y_pred_group,
        zero_division=0
    )

    product_results.append(
        (category, recall, precision, mask.sum())
    )

    print(
        f"{category} | "
        f"Recall={recall:.4f} | "
        f"Precision={precision:.4f} | "
        f"N={mask.sum()}"
    )


# ------------------------------------------------------------
# Performance by PAYMENT METHOD
# ------------------------------------------------------------

print("\nPERFORMANCE BY PAYMENT_METHOD")

payment_results = []

for method in X_test["payment_method"].dropna().unique():

    mask = X_test["payment_method"] == method

    y_true_group = y_test.loc[mask]
    y_pred_group = y_test_pred[mask]

    recall = recall_score(
        y_true_group,
        y_pred_group,
        zero_division=0
    )

    precision = precision_score(
        y_true_group,
        y_pred_group,
        zero_division=0
    )

    payment_results.append(
        (method, recall, precision, mask.sum())
    )

    print(
        f"{method} | "
        f"Recall={recall:.4f} | "
        f"Precision={precision:.4f} | "
        f"N={mask.sum()}"
    )


# ------------------------------------------------------------
# Identify the worst subgroup by recall
# ------------------------------------------------------------

all_groups = []

for category, recall, precision, n in product_results:
    all_groups.append(
        ("product_category", category, recall, precision, n)
    )

for method, recall, precision, n in payment_results:
    all_groups.append(
        ("payment_method", method, recall, precision, n)
    )

worst_group = min(all_groups, key=lambda x: x[2])

print("\nWORST-PERFORMING SUBGROUP")
print(
    f"Group type: {worst_group[0]} | "
    f"Group: {worst_group[1]} | "
    f"Recall={worst_group[2]:.4f} | "
    f"Precision={worst_group[3]:.4f} | "
    f"N={worst_group[4]}"
)

print(
    "\nNEXT STEP: "
    f"Use a category-specific decision threshold or model calibration "
    f"for the {worst_group[0]}='{worst_group[1]}' subgroup "
    f"to improve its recall while monitoring precision."
)
# ============================================================
# TASK 9 - SAVE FINAL RANDOM FOREST ARTIFACT
# ============================================================

import os
import joblib
from sklearn.metrics import f1_score, recall_score, precision_score

# ------------------------------------------------------------
# Re-run threshold sweep using Random Forest probabilities
# ------------------------------------------------------------

rf_prob = best_rf.predict_proba(X_test)[:, 1]

best_rf_f1 = -1
t_star_rf = 0.5
best_rf_recall = 0
best_rf_precision = 0

print("\nRANDOM FOREST THRESHOLD SWEEP")

for threshold in [i / 100 for i in range(10, 91, 2)]:

    rf_pred = (rf_prob >= threshold).astype(int)

    f1 = f1_score(y_test, rf_pred, zero_division=0)
    recall = recall_score(y_test, rf_pred, zero_division=0)
    precision = precision_score(y_test, rf_pred, zero_division=0)

    print(
        f"Threshold={threshold:.2f} "
        f"F1={f1:.4f} "
        f"Recall={recall:.4f} "
        f"Precision={precision:.4f}"
    )

    if f1 > best_rf_f1:
        best_rf_f1 = f1
        t_star_rf = threshold
        best_rf_recall = recall
        best_rf_precision = precision


# ------------------------------------------------------------
# Final Random Forest threshold
# ------------------------------------------------------------

print("\nFINAL RANDOM FOREST THRESHOLD")
print(f"t*_rf: {t_star_rf:.2f}")
print(f"Best F1: {best_rf_f1:.4f}")
print(f"Recall at t*_rf: {best_rf_recall:.4f}")
print(f"Precision at t*_rf: {best_rf_precision:.4f}")


# ------------------------------------------------------------
# Save the final fitted Random Forest pipeline
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)

model_path = "models/return_risk_model.pkl"

joblib.dump(best_rf, model_path)

print("\nMODEL SAVED SUCCESSFULLY")
print(f"Saved to: {model_path}")