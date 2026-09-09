"""
features.py — turning raw landmarks into what the classifier actually sees.

This module is deliberately pure NumPy. It imports neither OpenCV nor MediaPipe, so
the whole feature pipeline can be tested on a machine with no camera and no vision
stack installed (see tests/test_pipeline.py).

Three jobs live here:

  1. NORMALISE one hand, so the same sign looks the same from any distance.
  2. PACK a frame's hands into a fixed-length vector, with each hand always in the
     same slot — see the note on handedness below, it matters more than it looks.
  3. TURN a recorded sequence into one feature vector, trimmed and time-normalised.

WHY HANDEDNESS MATTERS
    MediaPipe returns detected hands in no guaranteed order. Packing them in the
    order they arrive means the right hand can land in slot 0 on one frame and slot 1
    on the next, which makes a two-handed sign look like two different signs to the
    classifier. So we read `multi_handedness` and give each hand a fixed slot.

    Note the frames are mirrored before detection (so the preview feels like a
    mirror), which means MediaPipe's "Left"/"Right" are the mirror's, not the
    signer's. That is fine: recording and recognition mirror identically, so the
    slots stay consistent. It is consistency that matters here, not anatomy.

WHY TRIM AND RESAMPLE
    A sign made half a second late fills a different part of the 30-frame window, and
    a flattened window is position-sensitive: the same sign started later becomes a
    completely different feature vector. Trimming the empty head and tail and then
    resampling to a fixed length removes both the start-time offset and differences
    in signing speed.
"""

import numpy as np

FRAMES = 30              # frames per sample after resampling (~1s at 30fps)
HAND_FEATURES = 63       # 21 keypoints x 3 coordinates
FRAME_FEATURES = 126     # 2 hands x 63

# Bumped whenever the feature layout changes, so a model trained on the old layout is
# rejected loudly instead of producing silent nonsense.
FEATURE_VERSION = 2

# Length of the vector sequence_features() returns.
VECTOR_LENGTH = FRAMES * FRAME_FEATURES + FRAME_FEATURES + 1


def normalise(landmarks):
    """Make one hand's 21 keypoints comparable across signers and distances.

    Two signers making the same sign produce very different raw coordinates: one may
    be close to the camera, one far; one on the left of the frame, one on the right.
    The classifier must not see those differences.

      1. Translate — put the wrist at the origin, removing position in frame.
      2. Scale — divide by the largest distance from the wrist, removing the effect
         of distance from the camera and of hand size.

    Accepts either a MediaPipe landmark list or a plain sequence of (x, y, z).
    Returns a flat list of 63 floats.
    """
    raw = getattr(landmarks, "landmark", landmarks)
    pts = [(lm.x, lm.y, lm.z) if hasattr(lm, "x") else tuple(lm) for lm in raw]

    wrist = pts[0]
    shifted = [(x - wrist[0], y - wrist[1], z - wrist[2]) for x, y, z in pts]

    span = max((x * x + y * y + z * z) ** 0.5 for x, y, z in shifted)
    if span == 0:
        span = 1e-6

    flat = []
    for x, y, z in shifted:
        flat.extend([x / span, y / span, z / span])
    return flat


def _slot_for(handedness_entry):
    """Fixed slot for a hand: mirrored-left -> 0, mirrored-right -> 1, unknown -> None."""
    try:
        label = handedness_entry.classification[0].label
    except (AttributeError, IndexError, TypeError):
        return None
    if label == "Left":
        return 0
    if label == "Right":
        return 1
    return None


def frame_features(result):
    """Pack the hands found in one frame into a fixed-length vector.

    Each hand goes to the slot its handedness dictates, so a given hand always
    occupies the same 63 columns. Missing hands stay zero, which is also the signal
    Gate 0 in confidence.py uses to spot an empty frame.
    """
    vec = np.zeros(FRAME_FEATURES, dtype=np.float32)

    hands = getattr(result, "multi_hand_landmarks", None)
    if not hands:
        return vec

    handedness = getattr(result, "multi_handedness", None) or []
    taken = [False, False]

    for i, hand in enumerate(hands[:2]):
        slot = _slot_for(handedness[i]) if i < len(handedness) else None

        # No handedness, or both hands claimed the same side: fall back to whichever
        # slot is still free, so we never silently drop a hand.
        if slot is None or taken[slot]:
            slot = 0 if not taken[0] else 1

        taken[slot] = True
        vec[slot * HAND_FEATURES:(slot + 1) * HAND_FEATURES] = normalise(hand)

    return vec


def trim(sequence):
    """Drop the empty frames before and after the sign itself.

    Returns a (0, FRAME_FEATURES) array when no hand was seen at all.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    active = np.flatnonzero((sequence != 0).any(axis=1))
    if active.size == 0:
        return np.zeros((0, sequence.shape[1]), dtype=np.float32)
    return sequence[active[0]:active[-1] + 1]


def resample(sequence, n=FRAMES):
    """Stretch or squeeze a sequence to exactly n frames by linear interpolation.

    This is what makes a slow signer and a fast signer produce comparable vectors.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape[0] == 0:
        return np.zeros((n, FRAME_FEATURES), dtype=np.float32)
    if sequence.shape[0] == n:
        return sequence.astype(np.float32, copy=True)

    src = np.arange(sequence.shape[0], dtype=np.float64)
    dst = np.linspace(0.0, sequence.shape[0] - 1, num=n)

    out = np.empty((n, sequence.shape[1]), dtype=np.float32)
    for column in range(sequence.shape[1]):
        out[:, column] = np.interp(dst, src, sequence[:, column])
    return out


def sequence_features(sequence):
    """Turn one recorded sample into the single vector the model is trained on.

    Three parts, concatenated:
      - the trimmed, time-normalised pose sequence (shape, held over time)
      - per-coordinate total movement (how much each keypoint travelled)
      - the fraction of the original window in which a hand was visible

    The movement term is what lets the classifier separate signs that pass through
    similar hand shapes but move differently.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    aligned = resample(trim(sequence), FRAMES)

    motion = np.abs(np.diff(aligned, axis=0)).sum(axis=0)
    presence = float((sequence != 0).any(axis=1).mean()) if sequence.shape[0] else 0.0

    return np.concatenate([
        aligned.ravel(),
        motion.astype(np.float32),
        np.array([presence], dtype=np.float32),
    ]).astype(np.float32)
