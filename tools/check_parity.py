"""
check_parity.py — prove web/js/features.js computes exactly what src/features.py does.

Why this exists
    The desktop app and the browser build the same feature vector from the same
    landmarks, and a model trained by one is scored by the other. A vector built even
    slightly differently is not a worse input — it is a meaningless one. The columns
    stop meaning what they meant during training, and the classifier reports high
    confidence about nonsense. That is precisely the failure this project is built to
    prevent, arriving through a port to another language.

    So the two implementations are not reviewed side by side and assumed to agree.
    They are run on the same inputs and compared.

Run:
    python tools/check_parity.py

Needs numpy and Node.js. Exits non-zero on any disagreement.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import features as feat  # noqa: E402
from console import enable_unicode  # noqa: E402

# Anything at or below this is float32 rounding, not a difference in the maths.
TOLERANCE = 1e-5


def landmark_cases(rng):
    """Hands to push through normalise() and frame_features(): the awkward ones."""
    def hand(scale=1.0, offset=(0.0, 0.0, 0.0), flat=False):
        points = np.zeros((21, 3)) if flat else rng.normal(0, 0.1, (21, 3))
        points[0] = 0.0                                   # wrist at the local origin
        return [[float(p[0] * scale + offset[0]),
                 float(p[1] * scale + offset[1]),
                 float(p[2] * scale + offset[2])] for p in points]

    return [
        {"name": "one hand, left",
         "hands": [{"landmarks": hand(), "handedness": "Left"}]},
        {"name": "one hand, right",
         "hands": [{"landmarks": hand(), "handedness": "Right"}]},
        {"name": "two hands, in order",
         "hands": [{"landmarks": hand(), "handedness": "Left"},
                   {"landmarks": hand(), "handedness": "Right"}]},
        {"name": "two hands, reversed",
         "hands": [{"landmarks": hand(), "handedness": "Right"},
                   {"landmarks": hand(), "handedness": "Left"}]},
        {"name": "both claim the same side",
         "hands": [{"landmarks": hand(), "handedness": "Right"},
                   {"landmarks": hand(), "handedness": "Right"}]},
        {"name": "no handedness reported",
         "hands": [{"landmarks": hand(), "handedness": None},
                   {"landmarks": hand(), "handedness": None}]},
        {"name": "far and small",
         "hands": [{"landmarks": hand(scale=0.15, offset=(0.8, 0.7, 0.1)),
                    "handedness": "Left"}]},
        {"name": "degenerate: every point on the wrist",
         "hands": [{"landmarks": hand(flat=True), "handedness": "Left"}]},
        {"name": "no hands at all", "hands": []},
    ]


def sequence_cases(rng):
    """Sequences to push through sequence_features(): every shape trim/resample sees."""
    def burst(start, length, frames=feat.FRAMES, noise=0.02):
        seq = np.zeros((frames, feat.FRAME_FEATURES))
        base = rng.normal(0, 1, feat.FRAME_FEATURES)
        for step in range(length):
            if start + step >= frames:
                break
            envelope = 0.5 + 0.5 * np.sin(np.pi * step / max(length - 1, 1))
            seq[start + step] = base * envelope + rng.normal(0, noise,
                                                             feat.FRAME_FEATURES)
        return seq

    cases = [
        ("empty window", np.zeros((feat.FRAMES, feat.FRAME_FEATURES))),
        ("full window", rng.normal(0, 1, (feat.FRAMES, feat.FRAME_FEATURES))),
        ("single active frame", burst(10, 1)),
        ("two active frames", burst(4, 2)),
        ("starts at frame 0", burst(0, 20)),
        ("runs to the last frame", burst(10, 20)),
        ("one frame at the very end", burst(feat.FRAMES - 1, 1)),
        ("shorter than the window", burst(3, 8)),
        # Longer than one window: the browser's rolling buffer can hand over more
        # than FRAMES rows, so resample has to squeeze as well as stretch.
        ("longer than the window", rng.normal(0, 1, (47, feat.FRAME_FEATURES))),
        ("one row only", rng.normal(0, 1, (1, feat.FRAME_FEATURES))),
    ]
    for i in range(6):
        cases.append((f"random burst {i + 1}",
                      burst(int(rng.integers(0, 8)), int(rng.integers(5, 26)))))
    return [{"name": name, "sequence": seq.tolist()} for name, seq in cases]


def main():
    enable_unicode()
    rng = np.random.default_rng(20260909)

    landmarks = landmark_cases(rng)
    sequences = sequence_cases(rng)

    expected = {
        "frame_features": [
            feat.frame_features(_FakeResult(case["hands"])).tolist()
            for case in landmarks
        ],
        "sequence_features": [
            feat.sequence_features(np.array(case["sequence"], dtype=np.float32)).tolist()
            for case in sequences
        ],
        "constants": {
            "FRAMES": feat.FRAMES,
            "HAND_FEATURES": feat.HAND_FEATURES,
            "FRAME_FEATURES": feat.FRAME_FEATURES,
            "FEATURE_VERSION": feat.FEATURE_VERSION,
            "VECTOR_LENGTH": feat.VECTOR_LENGTH,
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "input.json").write_text(
            json.dumps({"landmarks": landmarks, "sequences": sequences}),
            encoding="utf-8")

        result = subprocess.run(
            ["node", str(ROOT / "tools" / "parity_runner.mjs"),
             str(tmp / "input.json"), str(tmp / "output.json")],
            capture_output=True, text=True, encoding="utf-8", errors="replace")

        if result.returncode != 0:
            print("Node failed:\n" + (result.stderr or result.stdout))
            return 1

        actual = json.loads((tmp / "output.json").read_text(encoding="utf-8"))

    failures = []

    for name, value in expected["constants"].items():
        if actual["constants"].get(name) != value:
            failures.append(f"constant {name}: python {value}, "
                            f"javascript {actual['constants'].get(name)}")

    def compare(kind, cases, mine, theirs):
        for case, a, b in zip(cases, mine, theirs):
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            if a.shape != b.shape:
                failures.append(f"{kind} / {case['name']}: shape {a.shape} vs {b.shape}")
                print(f"  FAIL  {kind:<18} {case['name']}")
                continue
            gap = float(np.abs(a - b).max()) if a.size else 0.0
            if gap > TOLERANCE:
                failures.append(f"{kind} / {case['name']}: largest difference {gap:.3e}")
                print(f"  FAIL  {kind:<18} {case['name']:<32} gap {gap:.3e}")
            else:
                print(f"  ok    {kind:<18} {case['name']:<32} gap {gap:.3e}")

    print(f"Comparing {len(landmarks)} landmark cases and {len(sequences)} sequences "
          f"against web/js/features.js\n")
    compare("frame_features", landmarks,
            expected["frame_features"], actual["frame_features"])
    compare("sequence_features", sequences,
            expected["sequence_features"], actual["sequence_features"])

    print()
    if failures:
        for line in failures:
            print("  " + line)
        print(f"\n{len(failures)} mismatch(es). The two pipelines do NOT agree.")
        return 1

    print("The Python and JavaScript feature pipelines agree.")
    return 0


class _FakeResult:
    """Shapes a web-style hand list the way features.frame_features expects it."""

    class _Classification:
        def __init__(self, label):
            self.label = label

    class _Handedness:
        def __init__(self, label):
            self.classification = [_FakeResult._Classification(label)]

    class _Hand:
        def __init__(self, points):
            self.landmark = [type("L", (), {"x": p[0], "y": p[1], "z": p[2]})()
                             for p in points]

    def __init__(self, hands):
        self.multi_hand_landmarks = [self._Hand(h["landmarks"]) for h in hands] or None
        labels = [h["handedness"] for h in hands]
        self.multi_handedness = ([self._Handedness(l) if l else None for l in labels]
                                 if any(labels) else None)


if __name__ == "__main__":
    sys.exit(main())
