/**
 * test_web.mjs — the web build's self-test.
 *
 * The browser has its own classifier and its own gates (see web/js/classifier.js), so
 * the guarantee the desktop asserts in tests/test_pipeline.py has to be asserted here
 * too, on this code. A refusal gate that exists in Python and is merely assumed in
 * JavaScript is not a refusal gate.
 *
 * Run:
 *     node tools/test_web.mjs
 *
 * No browser, no camera, no network — synthetic data only.
 */

import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const web = (file) => pathToFileURL(join(HERE, "..", "web", "js", file)).href;

const feat = await import(web("features.js"));
const clf = await import(web("classifier.js"));
const vocab = await import(web("vocabulary.js"));
const seg = await import(web("segment.js"));

/* --- a deterministic normal generator, so a failure is reproducible --- */

function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function normals(random) {
  return () => {
    // Box-Muller
    let u = 0;
    let v = 0;
    while (u === 0) u = random();
    while (v === 0) v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

const random = rng(20260909);
const gauss = normals(random);

function makeBase() {
  return Float64Array.from({ length: feat.FRAME_FEATURES }, () => gauss());
}

/** One recording of a sign: a movement starting at an arbitrary moment. */
function makeSample(base, { start = null, length = null, noise = 0.02, scale = 1 } = {}) {
  const frames = feat.FRAMES;
  const at = start ?? Math.floor(random() * 6);
  const span = length ?? 16 + Math.floor(random() * 8);

  const sequence = Array.from({ length: frames },
    () => new Float32Array(feat.FRAME_FEATURES));

  for (let step = 0; step < span; step++) {
    if (at + step >= frames) break;
    const envelope = 0.5 + 0.5 * Math.sin((Math.PI * step) / Math.max(span - 1, 1));
    for (let j = 0; j < feat.FRAME_FEATURES; j++) {
      sequence[at + step][j] = (base[j] * envelope + gauss() * noise) * scale;
    }
  }
  return sequence;
}

/* --- the checks --- */

const checks = [];
const check = (name, fn) => checks.push([name, fn]);

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function trainedModel(labels = 4, perLabel = 14) {
  const bases = {};
  const samples = [];
  for (let i = 0; i < labels; i++) {
    const label = `sign_${i}`;
    bases[label] = makeBase();
    for (let k = 0; k < perLabel; k++) {
      samples.push({
        label,
        vector: feat.sequenceFeatures(makeSample(bases[label])),
      });
    }
  }
  return { model: clf.fit(samples), bases, samples };
}

check("a model needs at least two signs", () => {
  const base = makeBase();
  const samples = Array.from({ length: 8 }, () => ({
    label: "only_one", vector: feat.sequenceFeatures(makeSample(base)),
  }));
  let threw = false;
  try { clf.fit(samples); } catch { threw = true; }
  assert(threw, "fitting one label should have been refused");
});

check("genuine signs are recognised", () => {
  const { model, bases } = trainedModel();
  for (const label of Object.keys(bases)) {
    const vector = feat.sequenceFeatures(makeSample(bases[label]));
    const decision = clf.decide(model, vector);
    assert(decision.accepted, `refused a genuine ${label}: ${decision.reason}`);
    assert(decision.label === label,
      `recognised ${decision.label}, expected ${label}`);
  }
});

check("an empty frame is refused", () => {
  const { model } = trainedModel();
  const empty = new Float32Array(feat.VECTOR_LENGTH);
  const decision = clf.decide(model, empty);
  assert(!decision.accepted, "accepted an empty frame");
  assert(decision.reason === "no hands detected", `reason was: ${decision.reason}`);
});

check("pure noise is refused", () => {
  const { model } = trainedModel();
  const noise = Float32Array.from({ length: feat.VECTOR_LENGTH }, () => gauss() * 6);
  const decision = clf.decide(model, noise);
  assert(!decision.accepted,
    `accepted pure noise as '${decision.label}' at ` +
    `${Math.round(decision.probability * 100)}%`);
});

check("a sign at half scale is refused", () => {
  const { model, bases } = trainedModel();
  const label = Object.keys(bases)[0];
  const vector = feat.sequenceFeatures(makeSample(bases[label], { scale: 0.5 }));
  const decision = clf.decide(model, vector);
  assert(!decision.accepted, `accepted half-scale input as ${decision.label}`);
});

check("one bad recording cannot switch the novelty gate off", () => {
  const { samples } = trainedModel();
  const clean = clf.fit(samples);

  const spoiled = samples.map((s, i) => (i === 0
    ? { label: s.label,
        vector: Float32Array.from({ length: feat.VECTOR_LENGTH }, () => gauss() * 40) }
    : s));
  const damaged = clf.fit(spoiled);

  const growth = damaged.limit / clean.limit;
  assert(growth < 3.0, `one outlier inflated the novelty limit ${growth.toFixed(1)}x`);
});

check("the reduced space is actually reduced", () => {
  const { model } = trainedModel();
  assert(model.components.length > 0, "no components were fitted");
  assert(model.components.length <= clf.NOVELTY_COMPONENTS,
    `kept ${model.components.length} components, cap is ${clf.NOVELTY_COMPONENTS}`);
  assert(model.components.length < feat.VECTOR_LENGTH,
    "distances are still being measured in the full feature space");
});

check("the vote never uses more neighbours than the rarest sign has", () => {
  const bases = { a: makeBase(), b: makeBase() };
  const samples = [];
  for (let k = 0; k < 12; k++) {
    samples.push({ label: "a", vector: feat.sequenceFeatures(makeSample(bases.a)) });
  }
  for (let k = 0; k < 3; k++) {
    samples.push({ label: "b", vector: feat.sequenceFeatures(makeSample(bases.b)) });
  }
  const model = clf.fit(samples);
  assert(model.neighbours <= 3,
    `k=${model.neighbours} with a class of only 3 samples: it can never win its vote`);
});

check("a refusal is never worded as a sign", () => {
  for (const language of Object.keys(vocab.LANGUAGES)) {
    const refusal = vocab.message("not_understood", language);
    const spoken = vocab.labels().map((l) => vocab.toText(l, language));
    assert(!spoken.includes(refusal),
      `the ${language} refusal is identical to a sign: ${refusal}`);
  }
  const overlap = Object.keys(vocab.MESSAGES)
    .filter((k) => k in vocab.VOCABULARY);
  assert(overlap.length === 0, `messages collide with signs: ${overlap}`);
});

check("every sign has every language, in the right script", () => {
  const arabicScript = (text) => /[؀-ۿ]/.test(text);
  for (const [label, entry] of Object.entries(vocab.VOCABULARY)) {
    for (const key of ["en", "ar", "ku", "ku_latin"]) {
      assert(entry[key] && entry[key].trim(), `sign '${label}' is missing ${key}`);
    }
    assert(arabicScript(entry.ku), `sign '${label}' has Kurdish in Latin script`);
    assert(arabicScript(entry.ar), `sign '${label}' has no Arabic`);
  }
});

/* --- segmentation: deciding when a sign starts and stops --- */

const HAND = Float32Array.from({ length: feat.FRAME_FEATURES }, (_, i) => (i % 7) + 1);
const NOTHING = new Float32Array(feat.FRAME_FEATURES);

/** A segmenter with a clock we control, so the cooldown does not need real waiting. */
function segmenter(handlers = {}) {
  const clock = { t: 0 };
  const s = new seg.Segmenter(handlers, () => clock.t);
  return { s, clock };
}

const feed = (s, frames) => frames.map(([vector, hands]) => s.push(vector, hands));

check("a hand passing through does not start a capture", () => {
  const { s } = segmenter();
  // Two frames is below ENTER_FRAMES, and detection flickers constantly.
  const out = feed(s, [[HAND, 1], [HAND, 1], [NOTHING, 0], [NOTHING, 0]]);
  assert(out.every((w) => w === null), "a two-frame flicker produced a window");
  assert(s.state === "idle", `state is ${s.state}, expected idle`);
});

check("a sign is captured once the hands leave", () => {
  const { s } = segmenter();
  let window = null;
  for (let i = 0; i < 20; i++) window ??= s.push(HAND, 1);
  assert(window === null, "emitted before the hands had left");
  for (let i = 0; i < seg.EXIT_FRAMES; i++) window ??= s.push(NOTHING, 0);
  assert(window !== null, "never emitted after the hands left");
  assert(window.length >= 20, `window held ${window.length} frames`);
});

check("a dropped hand mid-sign does not cut the sign in half", () => {
  const { s } = segmenter();
  let windows = 0;
  const emit = (w) => { if (w) windows++; };
  for (let i = 0; i < 10; i++) emit(s.push(HAND, 1));
  // The tracker loses the hand for a few frames, as it does when a hand turns
  // edge-on. Below EXIT_FRAMES, so the capture must survive it.
  for (let i = 0; i < seg.EXIT_FRAMES - 1; i++) emit(s.push(NOTHING, 0));
  for (let i = 0; i < 10; i++) emit(s.push(HAND, 1));
  assert(windows === 0, `a mid-sign dropout split the sign into ${windows + 1} pieces`);
});

check("the frames before the trigger are kept", () => {
  const { s } = segmenter();
  // Hands are already in frame while the segmenter is still confirming them; those
  // frames hold the handshape as it forms and must not be thrown away.
  let window = null;
  for (let i = 0; i < 12; i++) window ??= s.push(HAND, 1);
  for (let i = 0; i < seg.EXIT_FRAMES; i++) window ??= s.push(NOTHING, 0);
  assert(window.length > 12,
    `window has ${window.length} frames for 12 signing frames — no pre-roll kept`);
});

check("a window cannot grow without limit", () => {
  const { s } = segmenter();
  let window = null;
  for (let i = 0; i < seg.MAX_FRAMES * 3 && !window; i++) window = s.push(HAND, 1);
  assert(window !== null, "hands held up forever never produced a window");
  assert(window.length <= seg.MAX_FRAMES + seg.PREROLL,
    `window grew to ${window.length} frames`);
});

check("a flicker too short to be a sign is discarded, not classified", () => {
  const { s, clock } = segmenter();
  let tooShort = -1;
  s.handlers.onTooShort = (active) => { tooShort = active; };

  let window = null;
  for (let i = 0; i < 3; i++) window ??= s.push(HAND, 1);
  for (let i = 0; i < seg.EXIT_FRAMES; i++) { clock.t += 30; window ??= s.push(NOTHING, 0); }

  assert(window === null, "classified a flicker as a sign");
  assert(tooShort >= 0 && tooShort < seg.MIN_ACTIVE,
    `onTooShort reported ${tooShort} active frames`);
});

check("a second sign cannot start during the cooldown", () => {
  const { s, clock } = segmenter();
  let first = null;
  for (let i = 0; i < 20; i++) first ??= s.push(HAND, 1);
  for (let i = 0; i < seg.EXIT_FRAMES; i++) first ??= s.push(NOTHING, 0);
  assert(first !== null, "the first sign never emitted");

  // Immediately afterwards, hands back up: still inside the cooldown.
  for (let i = 0; i < 10; i++) {
    assert(s.push(HAND, 1) === null, "a sign was read during the cooldown");
  }
  assert(s.state === "cooldown", `state is ${s.state}`);

  clock.t += seg.COOLDOWN_MS + 1;
  for (let i = 0; i < seg.ENTER_FRAMES; i++) s.push(HAND, 1);
  assert(s.state === "capturing", "the segmenter never recovered after the cooldown");
});

/* --- runner --- */

console.log(`Running ${checks.length} checks — no browser, no camera.\n`);

let failed = 0;
for (const [name, fn] of checks) {
  try {
    fn();
    console.log(`  ok    ${name}`);
  } catch (error) {
    failed++;
    console.log(`  FAIL  ${name}\n          ${error.message}`);
  }
}

console.log();
if (failed) {
  console.log(`${failed} of ${checks.length} checks failed.`);
  process.exit(1);
}
console.log(`All ${checks.length} checks passed.`);
