/**
 * storage.js — the samples this browser has been taught, kept in this browser.
 *
 * IndexedDB rather than localStorage: one sample is a 30x126 array of floats, and a
 * useful vocabulary is a few hundred of them. localStorage is a synchronous string
 * store with a few megabytes of room; it would both block the render loop and run out.
 *
 * WHAT IS STORED IS THE RAW SEQUENCE, NOT THE FEATURE VECTOR
 *   Straight from src/record_dataset.py, and for the same reason: the feature
 *   pipeline will change, and recordings are expensive to collect again. Storing the
 *   derived vector would mean every improvement to features.js silently invalidated
 *   everything already taught — and invalidated it quietly, since old vectors still
 *   have the right shape and would simply mean something else.
 *
 * NOTHING HERE LEAVES THE DEVICE. There is no server to send it to; the site is
 * static files. That is a property of the design, not a promise in a policy: a
 * recording of a person's hands is sensitive, and the safest place for it is the
 * machine it was made on.
 */

const DB_NAME = "destdeng";
const DB_VERSION = 1;
const STORE = "samples";

let database = null;

function open() {
  if (database) return Promise.resolve(database);

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
        store.createIndex("label", "label", { unique: false });
      }
    };
    request.onsuccess = () => { database = request.result; resolve(database); };
    request.onerror = () => reject(request.error);
  });
}

function run(mode, work) {
  return open().then((db) => new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE, mode);
    const store = transaction.objectStore(STORE);
    let result;
    try {
      result = work(store);
    } catch (error) {
      reject(error);
      return;
    }
    transaction.oncomplete = () => resolve(result && result.result !== undefined
      ? result.result : result);
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  }));
}

/**
 * Save one recorded sign.
 *
 * `sequence` is an array of Float32Array rows. It is flattened to a single typed
 * array so IndexedDB stores one buffer instead of thirty, and so a sample that came
 * back from the database is the same shape as one that never left.
 */
export async function add(label, sequence) {
  const rows = sequence.length;
  const width = sequence[0]?.length ?? 0;
  const flat = new Float32Array(rows * width);
  sequence.forEach((row, i) => flat.set(row, i * width));

  return run("readwrite", (store) =>
    store.add({ label, rows, width, flat, at: Date.now() }));
}

/** Every sample, rebuilt into rows. */
export async function all() {
  const records = await run("readonly", (store) => store.getAll());
  return (records ?? []).map((record) => ({
    id: record.id,
    label: record.label,
    at: record.at,
    sequence: Array.from({ length: record.rows }, (_, i) =>
      record.flat.subarray(i * record.width, (i + 1) * record.width)),
  }));
}

/** How many samples exist per label, without rebuilding any of them. */
export async function counts() {
  const records = await run("readonly", (store) => store.getAll());
  const tally = new Map();
  for (const record of records ?? []) {
    tally.set(record.label, (tally.get(record.label) ?? 0) + 1);
  }
  return tally;
}

export async function removeLabel(label) {
  return run("readwrite", (store) => {
    const index = store.index("label");
    const request = index.openCursor(IDBKeyRange.only(label));
    request.onsuccess = () => {
      const cursor = request.result;
      if (cursor) { cursor.delete(); cursor.continue(); }
    };
  });
}

export async function clear() {
  return run("readwrite", (store) => store.clear());
}

/**
 * True when this browser can store anything at all.
 *
 * Private windows and locked-down browsers can refuse IndexedDB outright. Finding
 * that out when the first sample is saved means losing it; finding out at startup
 * means being able to say so.
 */
export async function available() {
  try {
    await open();
    return true;
  } catch {
    return false;
  }
}
