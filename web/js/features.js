/**
 * features.js — the feature pipeline, ported from src/features.py.
 *
 * THIS FILE MUST STAY NUMERICALLY IDENTICAL TO features.py.
 *
 * Not "equivalent in spirit" — identical. The desktop trainer and this browser can
 * exchange a model, and a feature vector built even slightly differently is not a
 * worse input to the classifier, it is a meaningless one: the model would score
 * columns that no longer mean what they meant during training and report high
 * confidence about it. That is the exact failure mode the whole project is built to
 * avoid, arriving through the back door.
 *
 * tools/check_parity.py compares this file's output against the Python original on
 * random sequences and fails on any difference beyond float32 rounding. Change one,
 * change the other, run it.
 *
 * The reasoning behind each step lives in src/features.py and is not repeated here.
 */

export const FRAMES = 30;
export const HAND_FEATURES = 63;      // 21 keypoints x 3 coordinates
export const FRAME_FEATURES = 126;    // 2 hands x 63
export const FEATURE_VERSION = 2;
export const VECTOR_LENGTH = FRAMES * FRAME_FEATURES + FRAME_FEATURES + 1;

/**
 * Make one hand's 21 keypoints comparable across signers and distances:
 * wrist to the origin, then scale by the largest distance from it.
 * `landmarks` is MediaPipe's array of {x, y, z}. Returns 63 numbers.
 */
export function normalise(landmarks) {
  const wrist = landmarks[0];
  const shifted = landmarks.map((p) => [p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]);

  let span = 0;
  for (const [x, y, z] of shifted) {
    const d = Math.sqrt(x * x + y * y + z * z);
    if (d > span) span = d;
  }
  if (span === 0) span = 1e-6;

  const flat = new Array(HAND_FEATURES);
  for (let i = 0; i < shifted.length; i++) {
    flat[i * 3] = shifted[i][0] / span;
    flat[i * 3 + 1] = shifted[i][1] / span;
    flat[i * 3 + 2] = shifted[i][2] / span;
  }
  return flat;
}

/**
 * Pack one frame's hands into a fixed-length vector, each hand in the slot its
 * handedness dictates. `hands` is [{landmarks, handedness}], handedness "Left" or
 * "Right" as MediaPipe reports it for the (already mirrored) frame.
 */
export function frameFeatures(hands) {
  const vec = new Float32Array(FRAME_FEATURES);
  if (!hands || hands.length === 0) return vec;

  const taken = [false, false];
  for (const hand of hands.slice(0, 2)) {
    let slot = hand.handedness === "Left" ? 0 : hand.handedness === "Right" ? 1 : null;

    // No handedness, or both hands claimed the same side: take whichever slot is
    // still free, so a hand is never silently dropped.
    if (slot === null || taken[slot]) slot = !taken[0] ? 0 : 1;

    taken[slot] = true;
    const values = normalise(hand.landmarks);
    for (let i = 0; i < HAND_FEATURES; i++) vec[slot * HAND_FEATURES + i] = values[i];
  }
  return vec;
}

/** Row indices of `sequence` from the first non-empty frame to the last. */
function activeRange(sequence) {
  let first = -1;
  let last = -1;
  for (let i = 0; i < sequence.length; i++) {
    let any = false;
    for (let j = 0; j < FRAME_FEATURES; j++) {
      if (sequence[i][j] !== 0) { any = true; break; }
    }
    if (any) {
      if (first === -1) first = i;
      last = i;
    }
  }
  return [first, last];
}

/** Drop the empty frames before and after the sign itself. */
export function trim(sequence) {
  const [first, last] = activeRange(sequence);
  if (first === -1) return [];
  return sequence.slice(first, last + 1);
}

/**
 * Stretch or squeeze to exactly n frames by linear interpolation.
 * Mirrors numpy.interp over src = 0..len-1 and dst = linspace(0, len-1, n).
 */
export function resample(sequence, n = FRAMES) {
  if (sequence.length === 0) {
    return Array.from({ length: n }, () => new Float32Array(FRAME_FEATURES));
  }
  if (sequence.length === n) return sequence.map((row) => Float32Array.from(row));

  const last = sequence.length - 1;
  const out = [];
  for (let k = 0; k < n; k++) {
    // linspace endpoints are exact; only the interior is computed, which is what
    // numpy does and what keeps the two implementations agreeing at the edges.
    const position = k === n - 1 ? last : (k * last) / (n - 1);
    const low = Math.floor(position);
    const high = Math.min(low + 1, last);
    const t = position - low;

    const row = new Float32Array(FRAME_FEATURES);
    for (let j = 0; j < FRAME_FEATURES; j++) {
      const a = sequence[low][j];
      row[j] = t === 0 ? a : a + (sequence[high][j] - a) * t;
    }
    out.push(row);
  }
  return out;
}

/**
 * One recorded sample -> the single vector the classifier sees:
 * the aligned pose sequence, then per-coordinate total movement, then the
 * fraction of the original window in which a hand was visible.
 */
export function sequenceFeatures(sequence) {
  const aligned = resample(trim(sequence), FRAMES);

  const motion = new Float32Array(FRAME_FEATURES);
  for (let i = 1; i < aligned.length; i++) {
    for (let j = 0; j < FRAME_FEATURES; j++) {
      motion[j] += Math.abs(aligned[i][j] - aligned[i - 1][j]);
    }
  }

  let visible = 0;
  for (const row of sequence) {
    for (let j = 0; j < FRAME_FEATURES; j++) {
      if (row[j] !== 0) { visible++; break; }
    }
  }
  const presence = sequence.length ? visible / sequence.length : 0;

  const out = new Float32Array(VECTOR_LENGTH);
  let at = 0;
  for (const row of aligned) {
    for (let j = 0; j < FRAME_FEATURES; j++) out[at++] = row[j];
  }
  for (let j = 0; j < FRAME_FEATURES; j++) out[at++] = motion[j];
  out[at] = presence;
  return out;
}

/** True when nothing at all was in frame — Gate 0 in confidence.js. */
export function isEmpty(vector) {
  for (let i = 0; i < vector.length; i++) if (vector[i] !== 0) return false;
  return true;
}
