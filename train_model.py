

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

FEATURES = ["ED_N", "ED_M", "S_N", "S_M", "q_N", "q_M", "D_MN"]


def load_data(path):
    df = pd.read_csv(path)
    X = df[FEATURES].values
    y = df["label"].values
    return X, y, df


def evaluate(name, model, X_test, y_test, scaler=None):
    Xt = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(Xt)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n--- {name} ---")
    print("Performance Metrics:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "model": model, "scaler": scaler}


def main(csv_path, out_model="sybil_model.pkl"):
    X, y, df = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")
    print(f"Train Sybil ratio: {y_train.mean()*100:.1f}%   Test Sybil ratio: {y_test.mean()*100:.1f}%")

    results = []

    # 1) Logistic Regression (needs scaling, gives a clean linear baseline)
    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(scaler.transform(X_train), y_train)
    results.append(evaluate("Logistic Regression", lr, X_test, y_test, scaler))

    # 2) Random Forest (no scaling needed, robust default choice for tabular data)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    # 3) Gradient Boosting (usually squeezes out a bit more accuracy)
    gb = GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.08, random_state=42,
    )
    gb.fit(X_train, y_train)
    results.append(evaluate("Gradient Boosting", gb, X_test, y_test))

    best = max(results, key=lambda r: r["f1"])
    print(f"\n{'='*50}\nBEST MODEL: {best['name']}  (F1 = {best['f1']:.4f})\n{'='*50}")

    if best["name"] != "Logistic Regression":
        importances = best["model"].feature_importances_
        print("\nFeature importances:")
        for f, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
            print(f"  {f:6s}: {imp:.4f}")

    joblib.dump({"model": best["model"], "scaler": best["scaler"], "features": FEATURES,
                 "model_name": best["name"]}, out_model)
    print(f"\nSaved best model to: {out_model}")

    return best, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="sumo_features_dataset.csv")
    parser.add_argument("--out", type=str, default="sybil_model.pkl")
    args = parser.parse_args()
    main(args.csv, args.out)
