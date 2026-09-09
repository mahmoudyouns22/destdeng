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
import * as bundle from "./bundle.js";
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
  shipped: [],           // signs committed alongside the site; never edited here
  session: null,         // a guided run through the whole vocabulary
  lastReading: null,     // { accepted, key, detail, probability }
  counts: new Map(),
};

const el = (id) => document.getElementById(id);

/**
 * Attach a listener, and do nothing if that element is not on the page.
 *
 * Not defensiveness for its own sake. Every file here is cached independently by the
 * host — GitHub Pages sends max-age=600 on each one with no revalidation — so for a
 * few minutes after a deploy a visitor can hold a fresh index.html with a stale
 * app.js, or the reverse. The reverse is the dangerous one: new code reaching for a
 * button the old markup does not have would throw inside setup and take the whole
 * page down, turning a ten-minute cosmetic skew into a ten-minute outage.
 *
 * With this, a missing element costs exactly the feature it belongs to.
 */
function on(id, event, handler) {
  const node = el(id);
  if (node) node.addEventListener(event, handler);
  return node;
}
const ui = {
  screens: {
    language: el("screen-language"),
    live: el("screen-live"),
    teach: el("screen-teach"),
    board: el("screen-board"),
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
  sessionIdle: el("session-idle"),
  sessionLive: el("session-live"),
  sessionStep: el("session-step"),
  sessionSign: el("session-sign"),
  sessionGloss: el("session-gloss"),
  sessionFill: el("session-fill"),
  sessionCount: el("session-count"),
  sessionAdvice: el("session-advice"),
  board: el("board"),
  boardLead: el("board-lead"),
  boardLangs: el("board-langs"),
  boardShow: el("board-show"),
  showWord: el("show-word"),
  showOther: el("show-other"),
  teachBar: el("teach-bar"),
  fault: el("fault"),
  faultTitle: el("fault-title"),
  faultDetail: el("fault-detail"),
};

function show(name) {
  state.screen = name;
  for (const [key, node] of Object.entries(ui.screens)) {
    if (node) node.hidden = key !== name;
  }
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
    // Two true things, in the order that helps: what does not work yet, and what
    // does. Saying only the first leaves someone at a desk with nothing.
    ui.panelText.textContent = state.model
      ? "Make a sign — the meaning appears here."
      : "Sign recognition is not trained yet — that work is under way.";
    ui.panelDetail.textContent = state.model
      ? `${state.counts.size} signs known`
      : "The word board works now. Tap here to open it.";
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
  for (const button of document.querySelectorAll("#langs button, #board-langs button")) {
    button.setAttribute("aria-pressed", String(button.dataset.language === language));
  }
  paintPanel();
  renderBoard();
  if (state.session) renderSession();
}

function buildLanguageButtons() {
  const native = { en: "EN", ar: "ع", ku: "کو" };
  for (const host of [ui.langs, ui.boardLangs]) {
    if (!host) continue;
    host.innerHTML = "";
    for (const [code, name] of Object.entries(LANGUAGES)) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.language = code;
      button.textContent = native[code] ?? code.toUpperCase();
      button.title = name;
      button.setAttribute("aria-label", name);
      button.addEventListener("click", () => setLanguage(code));
      host.append(button);
    }
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
    await store.add(label, window);

    if (state.session) {
      // Stay armed. Re-arming by hand after every sample is what turns teaching a
      // vocabulary into a chore nobody finishes; the segmenter's own cooldown is
      // already the pause between one sample and the next.
      state.session.done += 1;
      state.counts.set(label, (state.counts.get(label) ?? 0) + 1);
      renderTaughtList();

      if (state.session.done >= state.session.target) advanceSession();
      else renderSession();
      return;
    }

    state.teaching = null;
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

// How many samples one sign gets in a guided run. Twelve is a compromise found by
// what people will actually sit through: fewer and the novelty gate refuses honest
// signing, many more and the session is abandoned half-taught — which is worse than
// not starting, because a classifier can only ever answer with a sign it was shown,
// so every missing sign becomes a confident wrong one.
const SAMPLES_PER_SIGN = 12;

function advise(text, tone = "") {
  if (!ui.teachAdvice) return;
  ui.teachAdvice.textContent = text;
  ui.teachAdvice.dataset.tone = tone;
}

/** Redraw the list from state.counts, without touching the database. */
function renderTaughtList() {
  ui.teachList.innerHTML = "";

  if (state.counts.size === 0) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "Nothing taught yet";
    ui.teachList.append(empty);
    return;
  }

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

/**
 * Re-read what is stored and redraw.
 *
 * Reads the whole store, so it is called when something has changed structurally —
 * not after every sample during a session, where the count is already known and
 * re-reading every recording to count them would stall the capture loop.
 */
async function refreshTaught() {
  state.counts = await store.counts();
  for (const sample of state.shipped) {
    state.counts.set(sample.label, (state.counts.get(sample.label) ?? 0) + 1);
  }
  renderTaughtList();

  const signs = state.counts.size;
  const thin = [...state.counts].filter(([, n]) => n < 8).map(([l]) => l);

  if (signs < 2) {
    advise(`Teach at least two signs — a classifier given one thing to recognise ` +
           `has learned nothing. ${signs} taught so far.`);
  } else if (thin.length) {
    advise(`${thin.slice(0, 4).join(", ")}${thin.length > 4 ? "…" : ""} ` +
           `${thin.length === 1 ? "has" : "have"} fewer than 8 samples. It will work, ` +
           `but it will refuse more often than it needs to.`);
  } else {
    advise(`${signs} signs taught. More variety in the samples — distance, angle, ` +
           `lighting — is what makes it refuse less often.`, "ready");
  }
}

async function rebuildModel() {
  // Shipped signs and taught signs are one training set. Keeping them apart at fit
  // time would mean a visitor who teaches one sign of their own suddenly has a model
  // that knows only that sign.
  const samples = [...state.shipped, ...(await store.all())];
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
 * the guided session — teaching the whole vocabulary in one run        *
 * ------------------------------------------------------------------ */

function startSession() {
  if (!ui.sessionLive) return;
  state.session = {
    order: Object.keys(VOCABULARY),
    index: 0,
    target: SAMPLES_PER_SIGN,
    done: 0,
  };
  ui.sessionIdle.hidden = true;
  ui.sessionLive.hidden = false;
  settleSession();
}

/** Land on the next sign that still needs samples, or finish. */
function settleSession() {
  const session = state.session;
  if (!session) return;

  // Signs that already have enough are stepped over rather than recorded again, so a
  // session interrupted halfway resumes instead of starting from the beginning.
  while (session.index < session.order.length &&
         (state.counts.get(session.order[session.index]) ?? 0) >= session.target) {
    session.index += 1;
  }

  if (session.index >= session.order.length) return finishSession();

  const label = session.order[session.index];
  session.done = state.counts.get(label) ?? 0;
  state.teaching = label;
  segmenter.reset();
  renderSession();
}

function advanceSession() {
  if (!state.session) return;
  state.session.index += 1;
  settleSession();
}

async function finishSession() {
  state.session = null;
  state.teaching = null;
  if (ui.sessionIdle) ui.sessionIdle.hidden = false;
  if (ui.sessionLive) ui.sessionLive.hidden = true;

  setStatus(ui.teachStatus, "idle", "Fitting");
  await refreshTaught();
  await rebuildModel();
  setStatus(ui.teachStatus, state.model ? "live" : "warn",
            state.model ? "Ready" : "Not enough");

  if (state.model) {
    advise(`Done — ${state.counts.size} signs. Press “Done” to try it, and save the ` +
           `signs to a file so they survive this browser.`, "ready");
  }
}

function renderSession() {
  const session = state.session;
  if (!session || !ui.sessionStep) return;

  const label = session.order[session.index];
  const rtl = isRtl(state.language);

  ui.sessionStep.textContent =
    `Sign ${session.index + 1} of ${session.order.length}`;

  // The word is shown in the language chosen on the way in, because the person
  // teaching may not read English, and the label underneath is the key the model
  // actually uses — the two are different things and are shown as different things.
  ui.sessionSign.textContent = toText(label, state.language);
  ui.sessionSign.lang = state.language === "ku" ? "ckb" : state.language;
  ui.sessionSign.dir = rtl ? "rtl" : "ltr";
  ui.sessionSign.classList.toggle("is-arabic", rtl);
  ui.sessionGloss.textContent = label;

  const fraction = Math.min(1, session.done / session.target);
  ui.sessionFill.style.width = `${Math.round(fraction * 100)}%`;
  ui.sessionCount.textContent = `${session.done} / ${session.target} samples`;

  ui.sessionAdvice.textContent =
    "Make the sign, then lower your hands. It records by itself and starts the next " +
    "one. Change your distance and angle a little between samples — a model taught " +
    "from one exact pose refuses everything that is not it.";
}



/* ------------------------------------------------------------------ *
 * the word board — the part that works before anything is taught       *
 * ------------------------------------------------------------------ */

/**
 * Every sign as a word the reader can be shown directly.
 *
 * Recognition needs a trained model, and there is no public dataset for Kurdish Sign
 * Language to build one from. This needs nothing, and it addresses the same problem:
 * a deaf person and a clerk who read different scripts, with no interpreter in the
 * building. Tap a word, turn the screen around. People already do this with pen and
 * paper; the difference is that the reader gets it in the script they actually read,
 * and the writer does not have to know that script.
 *
 * MESSAGES are deliberately absent. "I did not understand" is something the SYSTEM
 * says about itself, and a person tapping it would be saying something they did not
 * mean. The separation is the same one src/vocabulary.py keeps for the same reason.
 */
function renderBoard() {
  if (!ui.board) return;

  const rtl = isRtl(state.language);
  ui.board.setAttribute("dir", rtl ? "rtl" : "ltr");
  ui.board.lang = state.language === "ku" ? "ckb" : state.language;
  ui.board.classList.toggle("is-arabic", rtl);

  ui.boardLead.textContent =
    "Every word in all three languages. Tap one to show it large, then turn the " +
    "screen around. Nothing here needs the camera or any teaching.";

  ui.board.innerHTML = "";
  for (const label of Object.keys(VOCABULARY)) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "word";

    const main = document.createElement("span");
    main.className = "word__main";
    main.textContent = toText(label, state.language);

    // Every card carries all three languages, not just the chosen one.
    //
    // The chosen language is only a guess about who is reading. In a clinic the
    // person who walks up next reads something else, and a board that shows one
    // language at a time makes them wait while somebody finds a setting. Showing all
    // three costs one line of small text and removes the setting from the exchange
    // entirely: whoever is standing there finds their own script and reads it.
    const others = document.createElement("span");
    others.className = "word__others";
    for (const code of Object.keys(LANGUAGES)) {
      if (code === state.language) continue;
      const piece = document.createElement("span");
      piece.textContent = toText(label, code);
      piece.lang = code === "ku" ? "ckb" : code;
      piece.dir = isRtl(code) ? "rtl" : "ltr";
      if (isRtl(code)) piece.classList.add("is-arabic");
      others.append(piece);
    }

    card.append(main, others);
    card.addEventListener("click", () => showWord(label));
    ui.board.append(card);
  }
}

function showWord(label) {
  if (!ui.boardShow) return;
  const rtl = isRtl(state.language);

  ui.showWord.textContent = toText(label, state.language);
  ui.showWord.lang = state.language === "ku" ? "ckb" : state.language;
  ui.showWord.dir = rtl ? "rtl" : "ltr";
  ui.showWord.classList.toggle("is-arabic", rtl);

  // The other two languages underneath, smaller. Which language the reader actually
  // reads is not always the one that was chosen at the door, and a second person
  // arriving should not need anyone to change a setting.
  const others = Object.keys(LANGUAGES)
    .filter((code) => code !== state.language)
    .map((code) => toText(label, code));
  ui.showOther.textContent = others.join("   ·   ");

  ui.boardShow.hidden = false;
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

  const shipped = await bundle.loadShipped();
  state.shipped = shipped.samples;
  if (shipped.problem) {
    // A signs file that is present but unreadable must not look identical to no
    // signs file at all: one is a normal deployment, the other is a broken one.
    console.warn("shipped signs ignored:", shipped.problem);
  }

  await refreshTaught();
  await rebuildModel();
  paintPanel();
  renderBoard();
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

  on("btn-board", "click", () => {
    renderBoard();
    show("board");
  });

  on("btn-board-back", "click", () => show("live"));

  on("board-show", "click", () => { ui.boardShow.hidden = true; });

  on("btn-teach", "click", async () => {
    show("teach");
    await refreshTaught();
    setStatus(ui.teachStatus, "idle", "Ready");
  });

  on("btn-back", "click", async () => {
    if (state.session) await finishSession();
    state.teaching = null;
    show("live");
    await rebuildModel();
  });

  on("btn-session", "click", startSession);
  on("btn-skip", "click", advanceSession);
  on("btn-stop", "click", finishSession);

  on("btn-save", "click", async () => {
    const taught = await store.all();
    if (taught.length === 0) {
      advise("Nothing taught on this device yet — there is nothing to save.");
      return;
    }
    bundle.save(taught, "destdeng-signs.json");
    advise(`Saved ${taught.length} samples. Commit that file to the repository as ` +
           `web/model/signs.json and everyone who opens the site gets them.`, "ready");
  });

  on("file-load", "change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const loaded = await bundle.read(file);
      for (const sample of loaded) await store.add(sample.label, sample.sequence);
      await refreshTaught();
      await rebuildModel();
      advise(`Loaded ${loaded.length} samples from ${file.name}.`, "ready");
    } catch (error) {
      // A rejected file is reported rather than half-loaded: a bundle from an older
      // feature layout would score old measurements as new ones.
      advise(`Could not load that file. ${error.message}`);
    }
    event.target.value = "";
  });

  on("btn-record", "click", () => {
    state.teaching = ui.teachLabel.value;
    segmenter.reset();
    advise(`Waiting for “${state.teaching}”. Make the sign now — it records by ` +
           `itself and stops when your hands leave the frame.`);
    setStatus(ui.teachStatus, "warn", "Show your hands");
  });

  on("btn-forget", "click", async () => {
    const total = [...state.counts.values()].reduce((a, b) => a + b, 0);
    if (!total) return;
    if (!confirm(`Delete all ${total} samples taught on this device? ` +
                 `This cannot be undone.`)) return;
    await store.clear();
    await refreshTaught();
    await rebuildModel();
  });

  on("fault-retry", "click", () => location.reload());

  // The panel is the obvious place to press when nothing has been taught, and on a
  // narrow screen the header button is hidden.
  // Tapping the panel when nothing is taught opens the board rather than the teaching
  // screen: it is the thing that works right now, and a screen that cannot answer
  // should hand over to one that can.
  on("panel", "click", () => {
    if (state.model) return;
    renderBoard();
    show("board");
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
