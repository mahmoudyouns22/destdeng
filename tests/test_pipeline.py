"""
test_pipeline.py — check the parts of DESTDENG that do not need a camera.

Everything from normalisation to the refusal gates is exercised here on synthetic
data, so the pipeline can be verified on a machine with no webcam, no MediaPipe and no
OpenCV installed. That is deliberate: the vision stack is the hardest part of this
project to install and the least likely part to be wrong, and until now nothing at all
could be checked without it.

Run:
    python tests/test_pipeline.py

Needs only numpy and scikit-learn.
"""

import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import confidence                                        # noqa: E402
import features as feat                                  # noqa: E402
import vocabulary                                        # noqa: E402
from console import enable_unicode                       # noqa: E402


# --- fakes standing in for MediaPipe -----------------------------------------

class FakeLandmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class FakeHand:
    def __init__(self, points):
        self.landmark = [FakeLandmark(*p) for p in points]


class FakeClassification:
    def __init__(self, label):
        self.label = label


class FakeHandedness:
    def __init__(self, label):
        self.classification = [FakeClassification(label)]


class FakeResult:
    def __init__(self, hands=None, handedness=None):
        self.multi_hand_landmarks = hands
        self.multi_handedness = handedness


def a_hand(rng, offset=(0.0, 0.0, 0.0), scale=1.0):
    points = rng.normal(0, 0.1, (21, 3))
    points[0] = 0.0                                   # wrist at the local origin
    return FakeHand([(p[0] * scale + offset[0],
                      p[1] * scale + offset[1],
                      p[2] * scale + offset[2]) for p in points])


# --- synthetic dataset --------------------------------------------------------

def make_sample(rng, base, frames=feat.FRAMES, noise=0.02, start=None, length=None):
    """One recording of a sign: a movement that starts at an arbitrary moment."""
    sequence = np.zeros((frames, feat.FRAME_FEATURES), dtype=np.float32)
    start = rng.integers(0, 6) if start is None else start
    length = rng.integers(16, 24) if length is None else length

    for step in range(length):
        if start + step >= frames:
            break
        progress = step / max(length - 1, 1)
        envelope = 0.5 + 0.5 * np.sin(np.pi * progress)
        sequence[start + step] = base * envelope + rng.normal(0, noise, feat.FRAME_FEATURES)

    return sequence


def make_dataset(rng, n_labels=4, per_label=14):
    bases = {f"sign_{i}": rng.normal(0, 1, feat.FRAME_FEATURES) for i in range(n_labels)}
    X, y = [], []
    for label, base in bases.items():
        for _ in range(per_label):
            X.append(feat.sequence_features(make_sample(rng, base)))
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y), bases


# --- the checks ---------------------------------------------------------------

def test_normalise_ignores_position_and_distance():
    """The same hand shape near and far, left and right, must look identical."""
    rng = np.random.default_rng(0)
    points = rng.normal(0, 0.1, (21, 3))
    points[0] = 0.0

    near_left = FakeHand([tuple(p) for p in points])
    far_right = FakeHand([(p[0] * 0.4 + 0.7, p[1] * 0.4 + 0.3, p[2] * 0.4) for p in points])

    a = np.array(feat.normalise(near_left))
    b = np.array(feat.normalise(far_right))

    assert np.allclose(a, b, atol=1e-5), "normalisation is not position/scale invariant"


def test_handedness_pins_each_hand_to_a_slot():
    """Detection order must not decide which half of the vector a hand lands in."""
    rng = np.random.default_rng(1)
    left, right = a_hand(rng), a_hand(rng)

    one_way = feat.frame_features(FakeResult(
        [left, right], [FakeHandedness("Left"), FakeHandedness("Right")]))
    other_way = feat.frame_features(FakeResult(
        [right, left], [FakeHandedness("Right"), FakeHandedness("Left")]))

    assert np.allclose(one_way, other_way), \
        "swapping detection order changed the feature vector"
    assert one_way[:feat.HAND_FEATURES].any(), "left slot empty"
    assert one_way[feat.HAND_FEATURES:].any(), "right slot empty"


def test_no_hand_is_ever_dropped():
    """Missing or duplicated handedness must still keep both hands."""
    rng = np.random.default_rng(2)
    two = [a_hand(rng), a_hand(rng)]

    without = feat.frame_features(FakeResult(two, None))
    duplicated = feat.frame_features(FakeResult(
        two, [FakeHandedness("Right"), FakeHandedness("Right")]))

    for name, vector in (("no handedness", without), ("duplicate handedness", duplicated)):
        assert vector[:feat.HAND_FEATURES].any(), f"{name}: slot 0 empty"
        assert vector[feat.HAND_FEATURES:].any(), f"{name}: slot 1 empty"


def test_timing_does_not_change_the_features():
    """A sign made later in the window is the same sign.

    This is the whole point of trimming and resampling. The raw flattened window —
    what the model used to be trained on — is included as a control: it changes a
    great deal, which is what made start time look like a difference between signs.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, feat.FRAME_FEATURES)

    early = make_sample(rng, base, noise=0.0, start=0, length=20)
    late = make_sample(rng, base, noise=0.0, start=8, length=20)

    aligned_gap = np.abs(feat.sequence_features(early) - feat.sequence_features(late)).max()
    raw_gap = np.abs(early.ravel() - late.ravel()).max()

    assert aligned_gap < 1e-5, f"time alignment failed (gap {aligned_gap:.4f})"
    assert raw_gap > 0.1, "control failed: raw windows should differ"


def test_empty_input_produces_an_empty_vector():
    empty = np.zeros((feat.FRAMES, feat.FRAME_FEATURES), dtype=np.float32)
    vector = feat.sequence_features(empty)

    assert vector.shape == (feat.VECTOR_LENGTH,), f"wrong length: {vector.shape}"
    assert not vector.any(), "an empty recording produced non-zero features"


def test_gates_accept_real_signs_and_refuse_everything_else():
    """The rule the project lives by: never assert something false with confidence."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(4)
    X, y, bases = make_dataset(rng)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
    novelty = confidence.fit_novelty(X_train)

    label = "sign_0"
    genuine = feat.sequence_features(make_sample(rng, bases[label])).reshape(1, -1)
    accepted = confidence.decide(model, novelty, genuine)
    assert accepted.accepted, f"refused a genuine sample: {accepted.reason}"
    assert accepted.label == label, f"recognised {accepted.label}, expected {label}"

    empty = np.zeros((1, feat.VECTOR_LENGTH), dtype=np.float32)
    blank = confidence.decide(model, novelty, empty)
    assert not blank.accepted, "accepted an empty frame"
    assert blank.reason == "no hands detected"

    noise = rng.normal(0, 6, (1, feat.VECTOR_LENGTH))
    nonsense = confidence.decide(model, novelty, noise)
    assert not nonsense.accepted, \
        f"accepted pure noise as '{nonsense.label}' at {nonsense.probability:.0%}"


def test_novelty_threshold_survives_one_bad_recording():
    """A single outlier must not be able to switch the novelty gate off.

    Keying the limit to the most isolated training sample is exactly how that happens,
    which is why the threshold is a quantile.
    """
    rng = np.random.default_rng(5)
    X, y, _ = make_dataset(rng)

    clean = confidence.fit_novelty(X)

    spoiled = X.copy()
    spoiled[0] = rng.normal(0, 40, feat.VECTOR_LENGTH)     # one wildly bad recording
    damaged = confidence.fit_novelty(spoiled)

    growth = confidence.novelty_limit(damaged) / confidence.novelty_limit(clean)
    assert growth < 3.0, f"one outlier inflated the novelty limit {growth:.1f}x"


def test_refusal_is_never_mistaken_for_a_sign():
    """A failure message must not read as something the signer said."""
    overlap = set(vocabulary.MESSAGES) & set(vocabulary.VOCABULARY)
    assert not overlap, f"system messages collide with signs: {sorted(overlap)}"

    for language in vocabulary.LANGUAGES:
        refusal = vocabulary.message("not_understood", language)
        spoken = {vocabulary.to_text(label, language) for label in vocabulary.labels()}
        assert refusal not in spoken, \
            f"the {language} refusal is identical to a sign: {refusal!r}"


def test_every_sign_has_every_language():
    required = set(vocabulary.LANGUAGES) | {"ku_latin"}

    for label, entry in vocabulary.VOCABULARY.items():
        missing = required - set(entry)
        assert not missing, f"sign '{label}' is missing {sorted(missing)}"
        assert all(entry[key].strip() for key in required), f"sign '{label}' has a blank"

    for key, entry in vocabulary.MESSAGES.items():
        missing = required - set(entry)
        assert not missing, f"message '{key}' is missing {sorted(missing)}"


class Skipped(Exception):
    """Raised by a check that needs something this machine does not have.

    Only for the text-rendering check below, which cannot run without the display
    stack. Everything else in this file runs anywhere, and must keep doing so.
    """


def test_every_character_shown_on_screen_can_actually_be_drawn():
    """No boxes. Whatever is drawn must be drawable by the font that will draw it.

    This is a regression test for a bug that was invisible from the code: shaping in
    arabic-reshaper's Kurdish mode emits PRIVATE USE codepoints for ە ڵ ێ, which no
    ordinary font contains, so Tahoma rendered تێنەگەیشتم as ت□نە□گە□یشتم — and the
    failure landed on the message the system shows when it has NOT understood, which
    is the one string a reader has no way to check against anything.

    Testing the raw letters cannot catch it: every candidate font contains all 42 of
    them. What reaches the font is the reshaper's output, so that is what is checked.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
        import text_render
        from PIL import ImageFont
    except Exception as error:
        raise Skipped(f"needs the display stack ({type(error).__name__})")

    if text_render.FONT_PATH is None:
        raise Skipped("no usable font on this machine")

    face = ImageFont.truetype(text_render.FONT_PATH, 40)

    for table in (vocabulary.VOCABULARY, vocabulary.MESSAGES):
        for label, entry in table.items():
            for language in ("ar", "ku"):
                drawn = text_render.shape(entry[language])

                private = [c for c in drawn if 0xE000 <= ord(c) <= 0xF8FF]
                assert not private, (
                    f"{label}/{language} shapes to private-use codepoints "
                    f"{[hex(ord(c)) for c in private]} — no ordinary font has them")

                missing = text_render._glyphless(face, drawn)
                assert not missing, (
                    f"{label}/{language} needs {missing!r}, which "
                    f"{os.path.basename(text_render.FONT_PATH)} cannot draw")


def test_kurdish_is_written_in_the_script_its_readers_use():
    """Sorani in Arabic script, not Latin transliteration. See vocabulary.py."""
    def arabic_script(text):
        return any("؀" <= character <= "ۿ" for character in text)

    for label, entry in vocabulary.VOCABULARY.items():
        assert arabic_script(entry["ku"]), \
            f"sign '{label}' has Kurdish in Latin script: {entry['ku']!r}"
        assert arabic_script(entry["ar"]), f"sign '{label}' has no Arabic: {entry['ar']!r}"


# --- runner -------------------------------------------------------------------

def main():
    enable_unicode()

    checks = [value for name, value in sorted(globals().items())
              if name.startswith("test_") and callable(value)]

    print(f"Running {len(checks)} checks — no camera, no MediaPipe.\n")

    failures = []
    skipped = 0
    for check in checks:
        name = check.__name__[5:].replace("_", " ")
        try:
            check()
        except Skipped as why:
            # A skip is reported, never counted as a pass. The point of this file is
            # that it runs where the vision stack will not install, so one check that
            # genuinely needs it must say so out loud rather than quietly succeeding.
            print(f"  skip  {name}  ({why})")
            skipped += 1
        except Exception:
            print(f"  FAIL  {name}")
            failures.append((name, traceback.format_exc()))
        else:
            print(f"  ok    {name}")

    print()
    if failures:
        for name, detail in failures:
            print("=" * 70)
            print(f"FAILED: {name}")
            print("=" * 70)
            print(detail)
        print(f"{len(failures)} of {len(checks)} checks failed.")
        return 1

    passed = len(checks) - skipped
    if skipped:
        print(f"All {passed} checks passed, {skipped} skipped.")
    else:
        print(f"All {passed} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
