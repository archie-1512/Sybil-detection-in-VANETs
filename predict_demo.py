"""
Quick sanity-check harness: hand-crafted scenarios, same idea as the
"Running tests on various scenarios" table in the original slides.
"""
import joblib
import numpy as np
import pandas as pd

FEATURES = ["ED_N", "ED_M", "S_N", "S_M", "q_N", "q_M", "D_MN"]

# Hand-crafted feature vectors representing plausible scenarios.
# (ED_N, ED_M in metres; S_N, S_M in km/h; q_N, q_M = flow; D_MN = DTW distance)
SCENARIOS = [
    ("NORMAL - Free flowing traffic, independent vehicles",
     [70.0, 65.0, 45.0, 42.0, 900.0, 850.0, 1400.0], 0),
    ("NORMAL - Congested traffic, similarly slow but different patterns",
     [55.0, 60.0, 18.0, 22.0, 400.0, 480.0, 1800.0], 0),
    ("SYBIL - Close claimed positions, near-identical speed trace",
     [80.0, 79.0, 30.0, 30.5, 600.0, 610.0, 45.0], 1),
    ("SYBIL - Same physical vehicle broadcasting two identities",
     [50.0, 51.0, 20.0, 20.2, 300.0, 305.0, 12.0], 1),
]


def main(model_path="sybil_model.pkl"):
    bundle = joblib.load(model_path)
    model, scaler, model_name = bundle["model"], bundle["scaler"], bundle["model_name"]
    print(f"Model loaded successfully ({model_name})\n")
    print("Running tests on various scenarios...\n")
    header = f"{'Test Case':55s} {'Prediction':12s} {'Confidence':11s} {'Expected':10s}"
    print(header)
    print("-" * len(header))

    for desc, feats, expected in SCENARIOS:
        X = np.array([feats])
        Xt = scaler.transform(X) if scaler is not None else X
        proba = model.predict_proba(Xt)[0]
        pred = int(np.argmax(proba))
        conf = proba[pred]
        pred_label = "SYBIL" if pred == 1 else "NORMAL"
        expected_label = "SYBIL" if expected == 1 else "NORMAL"
        correct = "Correct" if pred == expected else "WRONG"
        print(f"{desc:55s} {pred_label:12s} {conf:<11.4f} {expected_label:10s} {correct}")


if __name__ == "__main__":
    main()
