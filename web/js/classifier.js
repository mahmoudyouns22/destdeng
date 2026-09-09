/**
 * classifier.js — learning signs in the browser, and refusing to answer.
 *
 * This is the web build's counterpart to src/train_model.py and src/confidence.py.
 * The rule it exists to enforce is the same one, and it is the reason the project
 * exists at all: a wrong translation in a hospital is worse than no translation, so
 * the system must be able to say "I did not understand".
 *
 * WHY THIS IS NOT THE SAME MODEL AS THE DESKTOP
 *   The desktop trains a Random Forest offline in Python. Training one in a browser
 *   is not practical, and requiring Python before the site does anything would defeat
 *   the point of a website. So the web build classifies by nearest neighbours in a
 *   reduced space instead — which is not a downgrade dressed up, it is the method the
 *   novelty gate already uses, now doing both jobs:
 *
 *     the distance to the nearest training sample answers "is this anything I know?"
 *     the vote among the k nearest answers "which sign is it?"
 *
 *   It also trains instantly, which is what makes teaching signs on the page possible.
 *   The cost is real and worth naming: a forest generalises better from few samples
 *   than nearest neighbours do, so expect the browser to need a few more samples per
 *   sign than the desktop to reach the same confidence.
 *
 * WHY THE DISTANCES ARE MEASURED IN A REDUCED SPACE
 *   Straight from src/confidence.py, and it is not an optimisation. In 3,907
 *   dimensions every point sits at roughly the same distance from every other, so
 *   "nearest neighbour" stops discriminating — the classifier would guess and the
 *   novelty gate would wave everything through while still appearing to be there.
 *   Standardising and projecting onto a few dozen components restores a distance that
 *   means something.
 *
 * WHY THE THRESHOLD IS A QUANTILE
 *   Also from confidence.py. Keying the novelty limit to the single most isolated
 *   training sample lets one bad recording inflate it far enough to admit anything.
 */

import { VECTOR_LENGTH, isEmpty } from "./features.js";

// Kept identical to src/confidence.py so the two builds refuse the same things.
export const PROBABILITY_THRESHOLD = 0.60;
export const NOVELTY_TOLERANCE = 1.6;
export const NOVELTY_QUANTILE = 0.95;
export const NOVELTY_COMPONENTS = 32;

// Neighbours voting on the label. Odd, and small because a sign may only have a
// handful of samples; capped again at fit time by the smallest class.
export const NEIGHBOURS = 5;


/* ------------------------------------------------------------------ *
 * linear algebra — only what is needed, no dependency                  *
 * ------------------------------------------------------------------ */

/**
 * Eigenvalues and eigenvectors of a small symmetric matrix, by cyclic Jacobi.
 *
 * Used on the Gram matrix, which is (samples x samples) — tens to low hundreds —
 * never on the (features x features) covariance, which would be 3907 x 3907 and
 * hopeless in a browser.
 *
 * Returns { values, vectors } with vectors in COLUMNS, sorted by descending value.
 */
function jacobiEigen(matrix, sweeps = 60) {
  const n = matrix.length;
  const a = matrix.map((row) => Float64Array.from(row));
  const v = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    row[i] = 1;
    return row;
  });

  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    }
    if (off < 1e-18) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;

        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;

        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const order = Array.from({ length: n }, (_, i) => i)
    .sort((i, j) => a[j][j] - a[i][i]);

  return {
    values: order.map((i) => a[i][i]),
    vectors: order.map((i) => Float64Array.from({ length: n }, (_, k) => v[k][i])),
  };
}


/* ------------------------------------------------------------------ *
 * the model                                                            *
 * ------------------------------------------------------------------ */

/**
 * Fit everything from the recorded samples.
 *
 * `samples` is [{ label, vector }] where vector came from sequenceFeatures().
 * Returns a model object, or throws with a message meant to be shown to a person.
 */
export function fit(samples) {
  const labels = [...new Set(samples.map((s) => s.label))].sort();
  if (labels.length < 2) {
    throw new Error("Needs at least two different signs — a classifier given one " +
                    "thing to recognise has learned nothing.");
  }

  const n = samples.length;
  const d = VECTOR_LENGTH;

  // --- standardise ------------------------------------------------------
  const mean = new Float64Array(d);
  for (const s of samples) for (let j = 0; j < d; j++) mean[j] += s.vector[j];
  for (let j = 0; j < d; j++) mean[j] /= n;

  const scale = new Float64Array(d);
  for (const s of samples) {
    for (let j = 0; j < d; j++) {
      const dev = s.vector[j] - mean[j];
      scale[j] += dev * dev;
    }
  }
  // A column that never varies carries no information and must not become a
  // division by zero; scikit-learn substitutes 1.0 for exactly this reason.
  for (let j = 0; j < d; j++) {
    const sd = Math.sqrt(scale[j] / n);
    scale[j] = sd > 1e-12 ? sd : 1;
  }

  const scaled = samples.map((s) => {
    const row = new Float64Array(d);
    for (let j = 0; j < d; j++) row[j] = (s.vector[j] - mean[j]) / scale[j];
    return row;
  });

  // --- PCA, via the Gram matrix ----------------------------------------
  // Eigenvectors of X'X (d x d, impossible here) are recovered from those of
  // XX' (n x n, trivial): if XX'v = λv then X'X(X'v) = λ(X'v).
  const gram = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let k = i; k < n; k++) {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += scaled[i][j] * scaled[k][j];
      gram[i][k] = gram[k][i] = dot / (n - 1);
    }
  }

  const { values, vectors } = jacobiEigen(gram);
  const wanted = Math.max(1, Math.min(NOVELTY_COMPONENTS, n - 1));

  const components = [];
  for (let c = 0; c < wanted; c++) {
    if (values[c] <= 1e-12) break;
    const direction = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      const weight = vectors[c][i];
      if (weight === 0) continue;
      for (let j = 0; j < d; j++) direction[j] += weight * scaled[i][j];
    }
    let norm = 0;
    for (let j = 0; j < d; j++) norm += direction[j] * direction[j];
    norm = Math.sqrt(norm);
    if (norm < 1e-12) break;
    for (let j = 0; j < d; j++) direction[j] /= norm;
    components.push(direction);
  }

  const model = {
    version: 2,                      // matches FEATURE_VERSION
    labels,
    mean,
    scale,
    components,
    points: null,
    pointLabels: samples.map((s) => s.label),
    neighbours: NEIGHBOURS,
    limit: 0,
    median: 0,
  };

  model.points = samples.map((s) => project(model, s.vector));

  // --- what a normal distance looks like --------------------------------
  const nearest = model.points.map((point, i) => {
    let best = Infinity;
    for (let k = 0; k < model.points.length; k++) {
      if (k === i) continue;
      const dist = distance(point, model.points[k]);
      if (dist < best) best = dist;
    }
    return best;
  }).filter(Number.isFinite).sort((a, b) => a - b);

  model.median = nearest[Math.floor(nearest.length / 2)] ?? 0;
  model.limit = quantile(nearest, NOVELTY_QUANTILE) * NOVELTY_TOLERANCE;

  // The smallest class caps k: voting with more neighbours than the rarest sign has
  // samples means that sign can never win its own vote.
  const smallest = Math.min(...labels.map(
    (l) => samples.filter((s) => s.label === l).length));
  model.neighbours = Math.max(1, Math.min(NEIGHBOURS, smallest));

  return model;
}

/** Linear interpolation between order statistics — numpy's default. */
function quantile(sorted, q) {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const position = q * (sorted.length - 1);
  const low = Math.floor(position);
  const high = Math.min(low + 1, sorted.length - 1);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

/** Standardise then project one feature vector into the reduced space. */
export function project(model, vector) {
  const d = vector.length;
  const scaled = new Float64Array(d);
  for (let j = 0; j < d; j++) scaled[j] = (vector[j] - model.mean[j]) / model.scale[j];

  const out = new Float64Array(model.components.length);
  for (let c = 0; c < model.components.length; c++) {
    const direction = model.components[c];
    let dot = 0;
    for (let j = 0; j < d; j++) dot += direction[j] * scaled[j];
    out[c] = dot;
  }
  return out;
}

function distance(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    const gap = a[i] - b[i];
    total += gap * gap;
  }
  return Math.sqrt(total);
}


/* ------------------------------------------------------------------ *
 * the decision                                                         *
 * ------------------------------------------------------------------ */

export class Decision {
  constructor(accepted, label, probability, distance, reason) {
    this.accepted = accepted;
    this.label = label;
    this.probability = probability;
    this.distance = distance;
    this.reason = reason;
  }
}

/**
 * Recognise one feature vector, or refuse to. Three gates, and all must pass.
 *
 * Gate 0  nothing was in frame
 * Gate 1  novelty — unlike anything trained on
 * Gate 2  probability — a known kind of input, but which sign is unclear
 */
export function decide(model, vector) {
  if (isEmpty(vector)) {
    return new Decision(false, null, 0, Infinity, "no hands detected");
  }

  const point = project(model, vector);

  const ranked = model.points
    .map((p, i) => ({ dist: distance(point, p), label: model.pointLabels[i] }))
    .sort((a, b) => a.dist - b.dist);

  const nearest = ranked[0].dist;

  // Gate 1 — novelty. Checked before the vote, because a vote among neighbours that
  // are all far away is a confident answer about nothing.
  if (nearest > model.limit) {
    return new Decision(false, ranked[0].label, 0, nearest,
      `unlike anything in training (distance ${nearest.toFixed(1)} > ` +
      `limit ${model.limit.toFixed(1)})`);
  }

  // Gate 2 — the vote, weighted by inverse distance so a neighbour sitting on top of
  // the sample counts for more than one at the edge of the accepted radius.
  const votes = new Map();
  let total = 0;
  for (const entry of ranked.slice(0, model.neighbours)) {
    const weight = 1 / (entry.dist + 1e-6);
    votes.set(entry.label, (votes.get(entry.label) ?? 0) + weight);
    total += weight;
  }

  let label = null;
  let best = 0;
  for (const [candidate, weight] of votes) {
    if (weight > best) { best = weight; label = candidate; }
  }
  const probability = total > 0 ? best / total : 0;

  if (probability < PROBABILITY_THRESHOLD) {
    return new Decision(false, label, probability, nearest,
      `ambiguous between known signs (${Math.round(probability * 100)}% < ` +
      `${Math.round(PROBABILITY_THRESHOLD * 100)}%)`);
  }

  return new Decision(true, label, probability, nearest, "accepted");
}


/* ------------------------------------------------------------------ *
 * storage                                                              *
 * ------------------------------------------------------------------ */

/**
 * A model is rebuilt from the samples rather than saved.
 *
 * Fitting takes well under a second for the sizes involved, and the samples are what
 * is worth keeping: the feature pipeline will change, and a saved model built on an
 * older layout would score new features against old columns and be confident about
 * nonsense. The desktop refuses such a model loudly (see load_model in
 * src/recognise.py); here there is simply nothing stale to load.
 */
export function serialiseSamples(samples) {
  return samples.map((s) => ({ label: s.label, sequence: s.sequence }));
}
