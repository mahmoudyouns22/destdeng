/**
 * segment.js — deciding, without being told, when a sign starts and when it ends.
 *
 * The desktop app does not need this: a person presses SPACE, and exactly 30 frames
 * are recorded. On a desk between two people that key is the problem. The signer
 * cannot reach it and the reader does not know when to press it, so the thing that
 * makes the desktop version tractable is the thing that makes it unusable in the
 * situation it was built for.
 *
 * So the web build watches instead. Hands enter the frame, move, and leave; that is
 * the whole signal, and it is enough for isolated signs.
 *
 * WHAT THIS IS NOT
 *   It is not continuous signing. It finds ONE sign at a time by looking for the gaps
 *   around it. Connected signed sentences have no such gaps — signs blend into each
 *   other and the grammar lives in the transitions — and reading them is an open
 *   research problem the README lists as post-V1. Nothing here solves it, and the
 *   interface says so rather than letting a fluent signer discover it.
 *
 * FOUR NUMBERS, AND WHY EACH IS NOT ZERO
 *
 *   ENTER  A single frame with a hand in it starts nothing. Detection flickers, and a
 *          hand reaching past the camera on its way somewhere else is not a sign.
 *
 *   EXIT   Nor does a single empty frame end a capture. The tracker drops a hand
 *          briefly all the time — mid-sign, when it turns edge-on to the lens — and
 *          ending there would cut the sign in half and file the halves as two signs.
 *
 *   PREROLL By the time ENTER frames have confirmed a hand is really there, the first
 *          moments of the sign have already gone past, and those are the frames where
 *          the handshape forms. So the last few frames before the trigger are kept
 *          and the capture starts from them.
 *
 *   MIN    A window with almost nothing in it is a flicker, not a sign. Classifying it
 *          would produce a confident answer about a hand that waved past, which is
 *          the failure this project exists to prevent.
 */

export const ENTER_FRAMES = 3;
export const EXIT_FRAMES = 6;
export const PREROLL = 4;
export const MAX_FRAMES = 48;
export const MIN_ACTIVE = 8;
export const COOLDOWN_MS = 700;

export class Segmenter {
  /**
   * `handlers` may provide onStart, onProgress(fraction), onEnd,
   * onWindow(frames) and onTooShort(activeCount).
   *
   * `now` is injectable so the cooldown can be tested without waiting for it.
   */
  constructor(handlers = {}, now = () => performance.now()) {
    this.handlers = handlers;
    this.now = now;
    this.reset();
  }

  reset() {
    this.state = "idle";
    this.buffer = [];
    this.preroll = [];
    this.present = 0;
    this.absent = 0;
    this.until = 0;
  }

  /**
   * Offer one frame.
   *
   * `vector` is frameFeatures() output for this frame; `hands` is how many hands were
   * seen in it. Returns the emitted window when this frame completed a sign, else null
   * — the handlers are for the interface, the return value is for tests.
   */
  push(vector, hands) {
    const now = this.now();

    if (this.state === "cooldown") {
      if (now < this.until) return null;
      this.state = "idle";
      this.present = 0;
    }

    if (this.state === "idle") {
      this.preroll.push(vector);
      if (this.preroll.length > PREROLL) this.preroll.shift();

      this.present = hands ? this.present + 1 : 0;
      if (this.present >= ENTER_FRAMES) {
        this.state = "capturing";
        this.buffer = this.preroll.slice();
        this.preroll = [];
        this.absent = 0;
        this.handlers.onStart?.();
      }
      return null;
    }

    this.buffer.push(vector);
    this.absent = hands ? 0 : this.absent + 1;
    this.handlers.onProgress?.(Math.min(1, this.buffer.length / MAX_FRAMES));

    if (this.absent < EXIT_FRAMES && this.buffer.length < MAX_FRAMES) return null;

    const window = this.buffer;
    this.buffer = [];
    this.state = "cooldown";
    this.until = now + COOLDOWN_MS;
    this.handlers.onEnd?.();

    const active = window.filter((row) => row.some((value) => value !== 0)).length;
    if (active < MIN_ACTIVE) {
      this.handlers.onTooShort?.(active);
      return null;
    }

    this.handlers.onWindow?.(window);
    return window;
  }
}
