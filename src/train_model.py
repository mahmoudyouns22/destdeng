"""
train_model.py — train the sign classifier on the recorded dataset.

Reads every .npy sample under data/<label>/, turns each one into a feature vector,
trains a classifier, reports how well it does on held-out samples, fits the novelty
check, and writes everything to models/sign_classifier.pkl.

Run:
    python src/train_model.py

The model is deliberately small. The dataset for Kurdish Sign Language has to be
recorded by hand, so there will never be a lot of it early on. A Random Forest over
normalised keypoints trains in seconds on a CPU and needs far less data than a deep
network over raw video would.

Samples are stored raw and featurised here, not at recording time. That is what lets
features.py change — better time alignment, new motion terms — without re-recording
people. The layout it produces is stamped into the saved file as FEATURE_VERSION, so a
model trained on an older layout is rejected loudly instead of quietly scoring noise.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import confidence
import features as feat
from console import enable_unicode

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sign_classifier.pkl"

MIN_SAMPLES_PER_LABEL = 5


def load_dataset(data_dir=DATA_DIR):
    """Load every recorded sample as a feature vector. Returns (X, y)."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise SystemExit(
            f"No dataset found at {data_dir}\n"
            f"Record some samples first:\n"
            f"    python src/record_dataset.py --label hello --samples 30"
        )

    samples, labels, skipped = [], [], []

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for npy in sorted(label_dir.glob("*.npy")):
            try:
                sequence = np.load(npy)
            except Exception as error:
                skipped.append(f"{npy.name}: unreadable ({type(error).__name__})")
                continue

            if sequence.ndim != 2 or sequence.shape[1] != feat.FRAME_FEATURES:
                skipped.append(
                    f"{npy.name}: expected (frames, {feat.FRAME_FEATURES}), "
                    f"got {sequence.shape}"
                )
                continue

            samples.append(feat.sequence_features(sequence))
            labels.append(label_dir.name)

    for problem in skipped:
        print(f"  skipped {problem}")

    if not samples:
        raise SystemExit(f"{data_dir} exists but holds no usable .npy samples.")

    return np.array(samples, dtype=np.float32), np.array(labels)


def train(X, y, test_size=0.25, seed=42):
    """Fit the classifier and the novelty check. Returns (bundle, report_text)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)

    report = classification_report(y_test, model.predict(X_test), zero_division=0)
    novelty = confidence.fit_novelty(X_train)

    bundle = {
        "model": model,
        "novelty": novelty,
        "feature_version": feat.FEATURE_VERSION,
        "labels": sorted(set(y.tolist())),
    }
    return bundle, report, len(X_train), len(X_test)


def main():
    enable_unicode()

    parser = argparse.ArgumentParser(description="Train the DESTDENG sign classifier.")
    parser.add_argument("--data", default=str(DATA_DIR),
                        help="dataset root folder (default: the project's data/)")
    parser.add_argument("--model", default=str(MODEL_PATH),
                        help="where to write the trained model")
    args = parser.parse_args()

    print("Loading dataset...")
    X, y = load_dataset(args.data)

    counts = Counter(y)
    print(f"\n{len(X)} samples across {len(counts)} signs "
          f"({X.shape[1]} features each):")
    for label, n in sorted(counts.items()):
        flag = "  <-- too few" if n < 10 else ""
        print(f"  {label:<14} {n:>4} samples{flag}")

    if len(counts) < 2:
        raise SystemExit(
            "\nNeed at least 2 different signs to train a classifier.\n"
            "Record another label and run this again."
        )

    thin = [label for label, n in counts.items() if n < MIN_SAMPLES_PER_LABEL]
    if thin:
        raise SystemExit(
            f"\nThese signs have fewer than {MIN_SAMPLES_PER_LABEL} samples: "
            f"{', '.join(sorted(thin))}\n"
            "Record more before training — a classifier trained on 2 examples has "
            "learned nothing."
        )

    print("\nTraining...")
    bundle, report, n_train, n_test = train(X, y)
    print(f"Trained on {n_train} samples, tested on {n_test}.")

    print("\n" + "=" * 68)
    print("RESULTS ON HELD-OUT SAMPLES")
    print("=" * 68)
    print(report)

    print("=" * 68)
    print("READ THIS BEFORE BELIEVING THE NUMBERS ABOVE")
    print("=" * 68)
    print(
        "These samples were held out at random, so the same signer almost certainly\n"
        "appears in both training and test. That measures very little.\n\n"
        "The number that matters is accuracy on a signer the model has NEVER seen.\n"
        "To measure it: record a new person, keep their samples out of training,\n"
        "and test on them alone. Expect the score to drop. That drop is the truth."
    )

    novelty = bundle["novelty"]
    print("\n" + "=" * 68)
    print("NOVELTY CHECK")
    print("=" * 68)
    print(f"  distances measured over {novelty['components']} components")
    print(f"  typical distance between training samples : "
          f"{novelty['median_train_distance']:.2f}")
    print(f"  {novelty['quantile']:.0%} of training samples sit within : "
          f"{novelty['reference_distance']:.2f}")
    print(f"  anything beyond {confidence.novelty_limit(novelty):.2f} "
          f"is treated as 'not a sign I know'")

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"\nModel saved to {model_path}")

    print(
        "\nNOTE: this file is derived from recordings of real people — the novelty\n"
        "check carries a compressed projection of the training set inside it. It is\n"
        "gitignored for that reason. Do not share it as though it were only weights."
    )
    print("\nNow run:  python src/recognise.py")


if __name__ == "__main__":
    main()
