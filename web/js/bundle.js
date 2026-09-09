/**
 * bundle.js — moving taught signs out of one browser and into another.
 *
 * Two problems, one file format.
 *
 * SIGNS TAUGHT IN A BROWSER LIVE IN THAT BROWSER
 *   IndexedDB is per-origin and per-device. Teaching fifteen signs is fifteen
 *   minutes of a person's time and it evaporates when they clear site data or open
 *   the page on their phone. So a session can be written to a file and read back.
 *
 * A VISITOR SHOULD NOT HAVE TO TEACH IT BEFORE IT DOES ANYTHING
 *   A recognition system that knows nothing until you spend a quarter of an hour on
 *   it will be closed in ten seconds. If `model/signs.json` is committed alongside
 *   the site, it is loaded at startup and the page works on arrival. Anything the
 *   visitor teaches is added on top; the shipped samples are never modified and
 *   "Forget everything" does not touch them.
 *
 * WHAT IS IN THE FILE, AND WHAT THAT MEANS
 *   Raw landmark sequences — the same thing src/record_dataset.py writes to disk,
 *   for the same reason: the feature pipeline will change, and recordings are
 *   expensive to collect again. Storing derived vectors instead would mean every
 *   improvement to features.js silently invalidated everything already taught, and
 *   invalidated it quietly, since old vectors still have the right shape.
 *
 *   These are landmark coordinates, not video — no image of anyone is in here. It is
 *   still derived from a person's hands, and publishing it publishes something of
 *   theirs. The desktop README makes the same point about the model file. Whoever
 *   commits one of these should be the person whose hands are in it, or should have
 *   asked.
 *
 * VERSION MISMATCHES FAIL LOUDLY
 *   A file written against an older feature layout is refused rather than loaded.
 *   Scoring old columns as though they were new ones does not produce worse answers,
 *   it produces confident meaningless ones — which is the single failure this
 *   project is built to prevent.
 */

import { FEATURE_VERSION, FRAME_FEATURES } from "./features.js";

export const FORMAT = "destdeng-signs/1";


/* --- Float32 <-> base64 ---------------------------------------------
 * JSON has no binary type, and a sequence written as decimal numbers is several
 * times larger than the bytes it represents. base64 of the raw Float32 buffer is
 * exact — no rounding on the way through — and compresses well in transit.
 */

function toBase64(floats) {
  const bytes = new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength);
  let binary = "";
  const CHUNK = 0x8000;              // avoid blowing the argument limit on apply()
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

function fromBase64(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}


/* --- writing --------------------------------------------------------- */

/** Turn stored samples into the object that gets written to a file. */
export function encode(samples, note = "") {
  const written = samples.map((sample) => {
    const rows = sample.sequence.length;
    const flat = new Float32Array(rows * FRAME_FEATURES);
    sample.sequence.forEach((row, i) => flat.set(row, i * FRAME_FEATURES));
    return { label: sample.label, rows, data: toBase64(flat) };
  });

  const counts = {};
  for (const sample of samples) {
    counts[sample.label] = (counts[sample.label] ?? 0) + 1;
  }

  return {
    format: FORMAT,
    featureVersion: FEATURE_VERSION,
    frameFeatures: FRAME_FEATURES,
    createdAt: new Date().toISOString(),
    note,
    // Not read back — a person opening the file should be able to see what is in it
    // without running anything.
    summary: { signs: Object.keys(counts).length, samples: samples.length, counts },
    samples: written,
  };
}

/** Offer the bundle to the visitor as a download. */
export function save(samples, filename = "destdeng-signs.json") {
  const blob = new Blob([JSON.stringify(encode(samples), null, 1)],
    { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();

  // Revoking immediately can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}


/* --- reading --------------------------------------------------------- */

/**
 * Validate and unpack a bundle. Throws with a message meant to be shown to a person.
 */
export function decode(object) {
  if (!object || object.format !== FORMAT) {
    throw new Error("That is not a DESTDENG signs file.");
  }
  if (object.featureVersion !== FEATURE_VERSION) {
    throw new Error(
      `This file was written for feature layout v${object.featureVersion}, but this ` +
      `version of the site produces v${FEATURE_VERSION}. Loading it would score old ` +
      `measurements as though they were new ones. Record the signs again.`);
  }
  if (object.frameFeatures !== FRAME_FEATURES) {
    throw new Error(
      `This file has ${object.frameFeatures} values per frame; this version expects ` +
      `${FRAME_FEATURES}.`);
  }
  if (!Array.isArray(object.samples) || object.samples.length === 0) {
    throw new Error("That file has no samples in it.");
  }

  return object.samples.map((sample, index) => {
    const flat = fromBase64(sample.data);
    const expected = sample.rows * FRAME_FEATURES;
    if (flat.length !== expected) {
      throw new Error(
        `Sample ${index + 1} (${sample.label}) says ${sample.rows} frames but holds ` +
        `${flat.length / FRAME_FEATURES}. The file is damaged.`);
    }
    return {
      label: sample.label,
      sequence: Array.from({ length: sample.rows },
        (_, i) => flat.subarray(i * FRAME_FEATURES, (i + 1) * FRAME_FEATURES)),
    };
  });
}

export async function read(file) {
  return decode(JSON.parse(await file.text()));
}

/**
 * The signs committed alongside the site, if there are any.
 *
 * Returns [] rather than throwing when the file is simply absent — a deployment
 * without one is a normal deployment, and the page says so and offers to be taught.
 * A file that IS there but cannot be read is a different matter and is reported,
 * because silently starting up knowing nothing would look identical to the case
 * above while meaning something quite different.
 */
export async function loadShipped(url = "model/signs.json") {
  let response;
  try {
    response = await fetch(url, { cache: "no-cache" });
  } catch {
    return { samples: [], problem: null };
  }
  if (!response.ok) return { samples: [], problem: null };

  try {
    return { samples: decode(await response.json()), problem: null };
  } catch (error) {
    return { samples: [], problem: error.message };
  }
}
