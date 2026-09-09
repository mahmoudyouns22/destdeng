/**
 * parity_runner.mjs — the JavaScript half of tools/check_parity.py.
 *
 * Reads the cases Python generated, runs them through the real web/js/features.js
 * (imported, not reimplemented — a copy here would only prove the copy agrees), and
 * writes the results back for Python to compare.
 *
 * Not used by the website itself.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));

// pathToFileURL, not the bare path: on Windows a dynamic import of "C:\..." is read
// as the URL scheme "c:" and rejected.
const {
  FRAMES, HAND_FEATURES, FRAME_FEATURES, FEATURE_VERSION, VECTOR_LENGTH,
  frameFeatures, sequenceFeatures,
} = await import(pathToFileURL(join(HERE, "..", "web", "js", "features.js")).href);

const [, , inputPath, outputPath] = process.argv;
const input = JSON.parse(readFileSync(inputPath, "utf-8"));

const frame_features = input.landmarks.map((testCase) =>
  Array.from(frameFeatures(testCase.hands.map((hand) => ({
    // Python is handed MediaPipe landmark objects; the web is handed the same shape.
    landmarks: hand.landmarks.map(([x, y, z]) => ({ x, y, z })),
    handedness: hand.handedness,
  }))))
);

const sequence_features = input.sequences.map((testCase) =>
  Array.from(sequenceFeatures(testCase.sequence.map((row) => Float32Array.from(row))))
);

writeFileSync(outputPath, JSON.stringify({
  constants: { FRAMES, HAND_FEATURES, FRAME_FEATURES, FEATURE_VERSION, VECTOR_LENGTH },
  frame_features,
  sequence_features,
}));
