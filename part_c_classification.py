"""Part C: delay classifier, feature signal, tuning, alert threshold."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cm_row(label, y_true, pred, **extra):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {label[0]: label[1], "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def f1_of(y_true, pred):
    return cm_row(("m", ""), y_true, pred)["f1"]


def write(rows, path, cols=None):
    out = pd.DataFrame(rows, columns=cols)
    num = out.select_dtypes(include="float").columns
    out[num] = out[num].round(4)
    out.to_csv(path, index=False)


# 2. Baseline and three classifiers (validation, 0.5 rule)
rows = [cm_row(("model", "blind_all_on_time"), y_val, np.zeros(len(y_val), dtype=int))]
logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
rows.append(cm_row(("model", "logistic_regression"), y_val, logit.predict(X_val)))
best_d, best_f1 = None, -1.0
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_of(y_val, t.predict(X_val))
    if f > best_f1:
        best_d, best_f1, best_tree = d, f, t
rows.append(cm_row(("model", f"decision_tree_depth_{best_d}"), y_val, best_tree.predict(X_val)))
rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append(cm_row(("model", "random_forest_default"), y_val, rf_default.predict(X_val)))
write(rows, "outputs/part_c_models.csv")

# 3. Feature signal
rf17 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
skb = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": skb.scores_, "rf_importance": rf17.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="min").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank", kind="stable")
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

k_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("sel", SelectKBest(f_classif, k=k)),
                  ("rf", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    k_rows.append({"k": k, "val_f1": f1_of(y_val, p.predict(X_val))})
write(k_rows, "outputs/part_c_selectk.csv")

# 4. Tuning on 7 selected features (selector fitted on training rows only)
sel7 = SelectKBest(f_classif, k=7).fit(X_train, y_train)
Xtr7, Xva7 = sel7.transform(X_train), sel7.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)
tun = [
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135, "best_params": gs.best_params_,
     "best_cv_accuracy": gs.best_score_, "val_f1": f1_of(y_val, gs.best_estimator_.predict(Xva7))},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45, "best_params": rs.best_params_,
     "best_cv_accuracy": rs.best_score_, "val_f1": f1_of(y_val, rs.best_estimator_.predict(Xva7))},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
     "best_cv_accuracy": np.nan, "val_f1": f1_of(y_val, default7.predict(Xva7))},
]
tun_out = pd.DataFrame(tun)
tun_out["best_params"] = tun_out["best_params"].map(lambda d: str(dict(d)))
tun_out[["best_cv_accuracy", "val_f1"]] = tun_out[["best_cv_accuracy", "val_f1"]].round(4)
tun_out.to_csv("outputs/part_c_tuning.csv", index=False)

# 5. Final configuration (ties -> default, grid, randomized)
priority = ["default_forest_same_7_features", "grid_search", "randomized_search"]
by_name = {r["search"]: r for r in tun}
chosen = max(priority, key=lambda n: (round(by_name[n]["val_f1"], 12), -priority.index(n)))
params = dict(by_name[chosen]["best_params"])
params.setdefault("max_depth", None)
final = Pipeline([("selector", SelectKBest(f_classif, k=7)),
                  ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_train, y_train)

# 6. Alert threshold from validation probabilities
proba_val = final.predict_proba(X_val)[:, 1]
thr_rows = [cm_row(("threshold", t), y_val, (proba_val >= t).astype(int))
            for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
write(thr_rows, "outputs/part_c_thresholds.csv")
ok = [r["threshold"] for r in thr_rows if r["recall"] >= 0.80]
threshold = max(ok) if ok else 0.1

# 7. One test evaluation
proba_test = final.predict_proba(X_test)[:, 1]
write([cm_row(("threshold", threshold), y_test, (proba_test >= threshold).astype(int))], "outputs/part_c_final_test.csv")
pd.DataFrame([{"selector_k": 7, "n_estimators": params["n_estimators"],
               "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
               "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold,
               "chosen_from": chosen}]).to_csv("outputs/part_c_final_config.csv", index=False)
print("chosen", chosen, params, "threshold", threshold)
