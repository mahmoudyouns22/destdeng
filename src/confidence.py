"""
confidence.py — deciding when NOT to answer.

This is the part of DESTDENG that matters most, so it lives in its own file.

A wrong translation in a hospital is worse than no translation. So the system has to be
able to say "I did not understand" — and that turns out to be harder than it sounds.

The obvious approach is to look at the classifier's probability and refuse below some
threshold. **That does not work on its own.** A Random Forest given input unlike
anything it was trained on still routes it down its trees and can come out 100%
confident about a sign the person never made. We measured this: feeding the trained
model pure noise produced a 100% confident answer.

So there are two independent gates, and a sample must pass both:

  1. NOVELTY — is this sample anything like the data we trained on?
     Measured as the distance to the nearest training sample. If the closest thing in
     the whole training set is still far away, we have never seen anything like this
     and we do not guess, whatever the classifier says.

  2. PROBABILITY — given that it IS a familiar kind of input, is the classifier
     actually sure which sign it is? This catches genuine ambiguity between two signs
     the model does know.

Gate 1 catches "that wasn't a sign at all". Gate 2 catches "that was a sign, but I
can't tell which". They fail in different ways, so both are needed.

TWO THINGS THE NOVELTY GATE HAS TO GET RIGHT

  Distances are measured in a reduced space, not on the raw 3,907 features. In a
  space that wide, every pair of points ends up at roughly the same distance from
  every other, so "nearest neighbour" stops meaning anything and the gate quietly
  stops working. Standardising and projecting onto a few dozen components restores a
  distance that actually discriminates.

  The threshold comes from a quantile, not from the maximum. Keying it to the single
  most isolated training sample lets one bad recording inflate the limit far enough
  to wave everything through — the gate would still be there, and it would still be
  useless.
"""

import numpy as np

# Minimum probability for the winning class, once novelty has passed.
PROBABILITY_THRESHOLD = 0.60

# How much further than a typical training sample an input may sit before we call it
# unfamiliar. 1.0 = as far as the reference distance below; higher is more lenient.
NOVELTY_TOLERANCE = 1.6

# Which quantile of the training nearest-neighbour distances sets the reference.
# 0.95 ignores the handful of most isolated samples without ignoring the real spread.
NOVELTY_QUANTILE = 0.95

# Components kept when reducing features for the distance measurement.
NOVELTY_COMPONENTS = 32


class Decision:
    """The outcome of one recognition attempt."""

    def __init__(self, accepted, label, probability, distance, reason):
        self.accepted = accepted
        self.label = label
        self.probability = probability
        self.distance = distance
        self.reason = reason

    def __repr__(self):
        state = "ACCEPT" if self.accepted else "REFUSE"
        return (f"<{state} {self.label} p={self.probability:.2f} "
                f"d={self.distance:.2f} ({self.reason})>")


def fit_novelty(X_train):
    """Learn what 'a normal distance from the training set' looks like.

    Features are standardised and projected onto a small number of components first
    (see the module docstring), then for each training sample we find its nearest
    OTHER training sample. The spread of those distances is what a new input is
    judged against.

    Returns a dict that gets saved alongside the classifier.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    X_train = np.asarray(X_train, dtype=np.float64)
    n_samples, n_features = X_train.shape

    scaler = StandardScaler().fit(X_train)
    scaled = scaler.transform(X_train)

    components = max(1, min(NOVELTY_COMPONENTS, n_samples - 1, n_features))
    pca = PCA(n_components=components, random_state=0).fit(scaled)
    reduced = pca.transform(scaled)

    neighbours = NearestNeighbors(n_neighbors=2).fit(reduced)
    distances, _ = neighbours.kneighbors(reduced)
    # column 0 is the sample itself (distance 0); column 1 is its nearest neighbour
    nearest = distances[:, 1]

    return {
        "scaler": scaler,
        "pca": pca,
        "nn": neighbours,
        "components": int(components),
        "quantile": NOVELTY_QUANTILE,
        "reference_distance": float(np.quantile(nearest, NOVELTY_QUANTILE)),
        "median_train_distance": float(np.median(nearest)),
        "max_train_distance": float(np.max(nearest)),
    }


def novelty_limit(novelty):
    """The distance beyond which an input counts as 'not a sign I know'."""
    return novelty["reference_distance"] * NOVELTY_TOLERANCE


def _project(novelty, features):
    return novelty["pca"].transform(novelty["scaler"].transform(features))


def decide(model, novelty, features):
    """Recognise one sample, or refuse to.

    `features` is the feature vector from features.sequence_features(), shaped
    (1, n_features).
    """
    features = np.asarray(features, dtype=np.float64)

    # Gate 0 — nothing was in frame at all.
    if not np.any(features):
        return Decision(False, None, 0.0, float("inf"),
                        "no hands detected")

    # Gate 1 — novelty.
    distance, _ = novelty["nn"].kneighbors(_project(novelty, features), n_neighbors=1)
    distance = float(distance[0][0])
    limit = novelty_limit(novelty)

    probabilities = model.predict_proba(features)[0]
    best = int(np.argmax(probabilities))
    label = model.classes_[best]
    probability = float(probabilities[best])

    if distance > limit:
        return Decision(False, label, probability, distance,
                        f"unlike anything in training (distance {distance:.1f} "
                        f"> limit {limit:.1f})")

    # Gate 2 — probability.
    if probability < PROBABILITY_THRESHOLD:
        return Decision(False, label, probability, distance,
                        f"ambiguous between known signs ({probability:.0%} "
                        f"< {PROBABILITY_THRESHOLD:.0%})")

    return Decision(True, label, probability, distance, "accepted")
