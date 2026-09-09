/**
 * app.js — DESTDENG in the browser.
 *
 * Camera in, text out, in the language the reader chose on the way in. Everything
 * runs on the device: the hand tracker is WebAssembly served from this site, the
 * classifier is fitted here, and the samples live in this browser. There is no
 * server, so there is nowhere for a video of someone's hands to go.
 *
 * THE ONE INTERACTION DECISION THAT MATTERS
 *   The desktop app reads a sign when you press SPACE. On a desk between two people
 *   that key is a problem: the signer cannot reach it and the reader does not know
 *   when to. So the web build watches continuously and decides for itself when a sign
 *   has started and finished — hands enter the frame, move, and leave.
 *
 *   Being honest about what that is: this recognises ONE SIGN AT A TIME, automatically.
 *   It is not continuous signing. Reading connected signed sentences — where signs
 *   blend into each other and grammar lives in the spaces between them — is an open
 *   research problem, listed as post-V1 in the README, and nothing here solves it.
 *   The difference matters to anyone deciding whether to rely on this.
 *
 *   How it decides is in segment.js, kept separate from this file so the state
 *   machine can be tested without a browser.
 *
 * THE SAME CAPTURE PATH IS USED FOR TEACHING AND FOR RECOGNISING
 *   Not a tidiness preference — a correctness requirement. One of the features is the
 *   fraction of the window in which a hand was visible. Teaching with a fixed
 *   countdown window and recognising with an automatically-trimmed one would give
 *   that feature two different distributions, and every genuine sign would be refused
 *   as unfamiliar. So teaching arms the same segmenter and stores what it emits.
 */

// Vendored, and named .js rather than .mjs on purpose: plenty of static servers
// still send .mjs as text/plain, and a module script served with the wrong content
// type is rejected outright with an error that says nothing about the cause.
import { HandLandmarker, FilesetResolver, DrawingUtils }
  from "../vendor/vision_bundle.js";
import { frameFeatures, sequenceFeatures } from "./features.js";
import { fit, decide } from "./classifier.js";
import * as store from "./storage.js";
import { Segmenter } from "./segment.js";
import { LANGUAGES, VOCABULARY, isRtl, message, toText } from "./vocabulary.js";

// No trailing slash: MediaPipe appends "/vision_wasm_internal.js" itself, and a
// doubled slash is tolerated by some hosts and 404s on others.
const VENDOR = new URL("../vendor", import.meta.url).href;

/* ------------------------------------------------------------------ *
 * application state                                                    *
 * ------------------------------------------------------------------ */

const state = {
  language: "en",
  model: null,
  landmarker: null,
  stream: null,
  screen: "language",
  teaching: null,        // label awaiting one sample, or null
  lastReading: null,     // { accepted, key, detail, probability }
  counts: new Map(),
};

const el = (id) => document.getElementById(id);
const ui = {
  screens: {
    language: el("screen-language"),
    live: el("screen-live"),
    teach: el("screen-teach"),
  },
  status: el("status"),
  langs: el("langs"),
  view: el("view"),
  captureBar: el("capture-bar"),
  hint: el("hint"),
  panel: el("panel"),
  panelText: el("panel-text"),
  panelDetail: el("panel-detail"),
  panelScore: el("panel-score"),
  panelPercent: el("panel-percent"),
  panelMeter: el("panel-meter"),
  teachView: el("teach-view"),
  teachStatus: el("teach-status"),
  teachLabel: el("teach-label"),
  teachAdvice: el("teach-advice"),
  teachList: el("teach-list"),
  teachBar: el("teach-bar"),
  fault: el("fault"),
  faultTitle: el("fault-title"),
  faultDetail: el("fault-detail"),
};

function show(name) {
  state.screen = name;
  for (const [key, node] of Object.entries(ui.screens)) node.hidden = key !== name;
}

function fault(title, detail) {
  ui.faultTitle.textContent = title;
  ui.faultDetail.textContent = detail;
  ui.fault.hidden = false;
}

function setStatus(node, tone, text) {
  node.dataset.state = tone;
  node.querySelector("b").textContent = text;
}


/* ------------------------------------------------------------------ *
 * the output panel                                                     *
 * ------------------------------------------------------------------ */

/**
 * Render the last reading in the currently selected language.
 *
 * The reading is held as a LABEL, never as rendered text, so switching language
 * re-renders what is already on screen. At a hospital desk the person pressing the
 * button is often not the person who has to read the word.
 */
function paintPanel() {
  const reading = state.lastReading;
  const rtl = isRtl(state.language);
  ui.panel.setAttribute("dir", rtl ? "rtl" : "ltr");
  ui.panel.lang = state.language === "ku" ? "ckb" : state.language;
  ui.panel.classList.toggle("is-arabic", rtl);

  if (!reading) {
    ui.panel.dataset.state = "empty";
    ui.panel.setAttribute("dir", "ltr");
    ui.panel.classList.remove("is-arabic");
    ui.panelText.textContent = state.model
      ? "Make a sign — the meaning appears here."
      : "This browser has not been taught any signs yet.";
    ui.panelDetail.textContent = state.model ? "" : "Open “Teach signs” to start.";
    ui.panelScore.hidden = true;
    return;
  }

  ui.panel.dataset.state = reading.accepted ? "accepted" : "refused";
  ui.panelText.textContent = reading.accepted
    ? toText(reading.key, state.language)
    : message(reading.key, state.language);
  ui.panelDetail.textContent = reading.detail;

  // The meter is drawn only for an accepted sign. A short bar beside a refusal
  // invites the reader to treat a rejected guess as a weak answer, and it is not an
  // answer at all.
  const scored = reading.accepted && reading.probability != null;
  ui.panelScore.hidden = !scored;
  if (scored) {
    ui.panelPercent.textContent = `${Math.round(reading.probability * 100)}%`;
    ui.panelMeter.style.width = `${Math.round(reading.probability * 100)}%`;
  }
}

function setLanguage(language) {
  state.language = language;
  for (const button of ui.langs.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.language === language));
  }
  paintPanel();
}

function buildLanguageButtons() {
  const native = { en: "EN", ar: "ع", ku: "کو" };
  ui.langs.innerHTML = "";
  for (const [code, name] of Object.entries(LANGUAGES)) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.language = code;
    button.textContent = native[code] ?? code.toUpperCase();
    button.title = name;
    button.setAttribute("aria-label", name);
    button.addEventListener("click", () => setLanguage(code));
    ui.langs.append(button);
  }
}


/* ------------------------------------------------------------------ *
 * camera and tracker                                                   *
 * ------------------------------------------------------------------ */

const video = document.createElement("video");
video.playsInline = true;
video.muted = true;

async function startTracker() {
  const fileset = await FilesetResolver.forVisionTasks(VENDOR);

  const options = {
    baseOptions: { modelAssetPath: `${VENDOR}/hand_landmarker.task`, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6,
  };

  try {
    return await HandLandmarker.createFromOptions(fileset, options);
  } catch {
    // No usable GPU path — older phones, locked-down browsers, some virtual
    // machines. CPU is slower but this has to work on the device people have.
    options.baseOptions.delegate = "CPU";
    return HandLandmarker.createFromOptions(fileset, options);
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser cannot open a camera from a web page.");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}


/* ------------------------------------------------------------------ *
 * the loop                                                             *
 * ------------------------------------------------------------------ */

let painter = null;
let lastTimestamp = -1;

const segmenter = new Segmenter({
  onStart() {
    if (state.screen === "live") {
      setStatus(ui.status, "reading", "Reading");
      ui.captureBar.hidden = false;
    } else {
      setStatus(ui.teachStatus, "reading", "Reading");
      ui.teachBar.hidden = false;
    }
  },
  onProgress(fraction) {
    const bar = state.screen === "live" ? ui.captureBar : ui.teachBar;
    bar.style.width = `${Math.round(fraction * 100)}%`;
  },
  onEnd() {
    ui.captureBar.hidden = true;
    ui.teachBar.hidden = true;
    ui.captureBar.style.width = "0%";
    ui.teachBar.style.width = "0%";
  },
  onWindow(window) { handleWindow(window); },
  onTooShort() {
    if (state.screen === "teach") {
      setStatus(ui.teachStatus, "warn", "Too brief");
      advise("That was too brief to be a sign. Hold the shape for about a second.");
    }
  },
});

async function handleWindow(window) {
  // Teaching: the same window that would have been recognised is stored instead.
  if (state.screen === "teach") {
    if (!state.teaching) return;
    const label = state.teaching;
    state.teaching = null;
    await store.add(label, window);
    await refreshTaught();
    await rebuildModel();
    setStatus(ui.teachStatus, "live", "Saved");
    advise(`Saved one “${label}”. Press record and make it again — vary your ` +
           `distance and angle a little each time.`);
    return;
  }

  if (!state.model) return;

  const decision = decide(state.model, sequenceFeatures(window));

  if (decision.accepted) {
    state.lastReading = {
      accepted: true,
      key: decision.label,
      detail: decision.label,
      probability: decision.probability,
    };
  } else {
    // Never assert something false with confidence. The wording is a system message,
    // so it can never be mistaken for the "repeat" sign the signer might have made.
    state.lastReading = {
      accepted: false,
      key: decision.reason === "no hands detected" ? "no_hands" : "not_understood",
      detail: decision.reason,
      probability: null,
    };
  }
  paintPanel();
}

function loop() {
  requestAnimationFrame(loop);

  const canvas = state.screen === "teach" ? ui.teachView : ui.view;
  if (!state.landmarker || video.readyState < 2 || state.screen === "language") return;

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    painter = null;
  }

  const context = canvas.getContext("2d");

  // Mirrored before detection, not only for display. The desktop build flips the
  // frame first too, so "Left" and "Right" mean the same thing in both — and a hand
  // must always land in the same half of the feature vector.
  context.save();
  context.scale(-1, 1);
  context.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  context.restore();

  const timestamp = performance.now();
  if (timestamp <= lastTimestamp) return;      // the tracker requires it to advance
  lastTimestamp = timestamp;

  let result;
  try {
    result = state.landmarker.detectForVideo(canvas, timestamp);
  } catch {
    return;
  }

  const hands = (result.landmarks ?? []).map((landmarks, i) => ({
    landmarks,
    handedness: result.handednesses?.[i]?.[0]?.categoryName ?? null,
  }));

  if (hands.length) {
    painter = painter ?? new DrawingUtils(context);
    for (const hand of hands) {
      painter.drawConnectors(hand.landmarks, HandLandmarker.HAND_CONNECTIONS,
        { color: "#4ECDC4", lineWidth: 3 });
      painter.drawLandmarks(hand.landmarks, { color: "#F5F5F7", radius: 3 });
    }
  }

  segmenter.push(frameFeatures(hands), hands.length);

  if (segmenter.state !== "capturing") {
    const node = state.screen === "teach" ? ui.teachStatus : ui.status;
    if (state.screen === "teach" && state.teaching) {
      setStatus(node, hands.length ? "live" : "warn",
        hands.length ? "Make the sign" : "Show your hands");
    } else if (hands.length) {
      setStatus(node, "live", hands.length === 2 ? "Tracking 2" : "Tracking 1");
    } else {
      setStatus(node, "warn", "No hands in frame");
    }
  }
}


/* ------------------------------------------------------------------ *
 * teaching                                                            *
 * ------------------------------------------------------------------ */

function advise(text, tone = "") {
  ui.teachAdvice.textContent = text;
  ui.teachAdvice.dataset.tone = tone;
}

async function refreshTaught() {
  state.counts = await store.counts();
  ui.teachList.innerHTML = "";

  if (state.counts.size === 0) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "Nothing taught yet";
    ui.teachList.append(empty);
  } else {
    for (const [label, count] of [...state.counts].sort()) {
      const row = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = `${label} — ${toText(label, state.language)}`;
      const tally = document.createElement("span");
      tally.className = count < 8 ? "count thin" : "count";
      tally.textContent = `${count}`;
      row.append(name, tally);
      ui.teachList.append(row);
    }
  }

  const signs = state.counts.size;
  const thin = [...state.counts].filter(([, n]) => n < 8).map(([l]) => l);

  if (signs < 2) {
    advise(`Teach at least two signs — a classifier given one thing to recognise ` +
           `has learned nothing. ${signs} taught so far.`);
  } else if (thin.length) {
    advise(`${thin.join(", ")} ${thin.length === 1 ? "has" : "have"} fewer than 8 ` +
           `samples. It will work, but it will refuse more often than it needs to.`);
  } else {
    advise("Enough to recognise. More samples, and more variety in them, is what " +
           "makes it refuse less often.", "ready");
  }
}

async function rebuildModel() {
  const samples = await store.all();
  const labels = new Set(samples.map((s) => s.label));

  if (labels.size < 2) {
    state.model = null;
    paintPanel();
    return;
  }

  try {
    state.model = fit(samples.map((s) => ({
      label: s.label,
      vector: sequenceFeatures(s.sequence),
    })));
  } catch {
    state.model = null;
  }
  paintPanel();
}

function buildLabelChoices() {
  ui.teachLabel.innerHTML = "";
  for (const label of Object.keys(VOCABULARY)) {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = `${label} — ${VOCABULARY[label].en}`;
    ui.teachLabel.append(option);
  }
}


/* ------------------------------------------------------------------ *
 * wiring                                                              *
 * ------------------------------------------------------------------ */

async function enter(language) {
  setLanguage(language);
  document.documentElement.lang = language === "ku" ? "ckb" : language;
  show("live");

  try {
    state.stream = await startCamera();
  } catch (error) {
    fault("The camera did not open",
      error?.name === "NotAllowedError"
        ? "Permission was refused. Allow camera access for this page in your " +
          "browser's address bar, then try again.\n\nNothing is recorded or " +
          "uploaded — the video is read on this device only."
        : `${error?.message ?? error}\n\nAnother app may be holding the camera.`);
    return;
  }

  setStatus(ui.status, "idle", "Loading tracker…");
  try {
    state.landmarker = await startTracker();
  } catch (error) {
    fault("The hand tracker did not load", String(error?.message ?? error));
    return;
  }

  await refreshTaught();
  await rebuildModel();
  paintPanel();
  loop();
}

function wire() {
  buildLanguageButtons();
  buildLabelChoices();

  for (const button of document.querySelectorAll("[data-language]")) {
    if (button.closest("#screen-language")) {
      button.addEventListener("click", () => enter(button.dataset.language));
    }
  }

  el("btn-teach").addEventListener("click", async () => {
    show("teach");
    await refreshTaught();
    setStatus(ui.teachStatus, "idle", "Ready");
  });

  el("btn-back").addEventListener("click", async () => {
    state.teaching = null;
    show("live");
    await rebuildModel();
  });

  el("btn-record").addEventListener("click", () => {
    state.teaching = ui.teachLabel.value;
    segmenter.reset();
    advise(`Waiting for “${state.teaching}”. Make the sign now — it records by ` +
           `itself and stops when your hands leave the frame.`);
    setStatus(ui.teachStatus, "warn", "Show your hands");
  });

  el("btn-forget").addEventListener("click", async () => {
    const total = [...state.counts.values()].reduce((a, b) => a + b, 0);
    if (!total) return;
    if (!confirm(`Delete all ${total} samples taught on this device? ` +
                 `This cannot be undone.`)) return;
    await store.clear();
    await refreshTaught();
    await rebuildModel();
  });

  el("fault-retry").addEventListener("click", () => location.reload());

  // The panel is the obvious place to press when nothing has been taught, and on a
  // narrow screen the header button is hidden.
  ui.panel.addEventListener("click", async () => {
    if (state.model) return;
    show("teach");
    await refreshTaught();
  });
}

async function start() {
  if (!(await store.available())) {
    fault("This browser will not let the page store anything",
      "Signs are taught and kept on your device, and that needs browser storage. " +
      "A private window, or a browser set to block site data, will prevent it.\n\n" +
      "Try a normal window.");
    return;
  }
  wire();
  paintPanel();
}

start();
