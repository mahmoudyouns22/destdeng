"""
record_dataset.py — record labelled sign samples for the DESTDENG dataset.

No usable dataset exists for Kurdish Sign Language, so it has to be recorded. This tool
records a short sequence of hand keypoints per sample and writes it to disk as a .npy
array, organised by sign label.

    data/
      hello/
        hello_1712345678.npy
        ...
      thank_you/
        ...

Each file holds an array of shape (FRAMES, 126):
  FRAMES  = frames captured per sample
  126     = 2 hands x 21 keypoints x 3 coordinates
Missing hands are recorded as zeros, so every sample has the same shape. Which hand
lands in which half of the 126 is decided by handedness, not detection order — see
src/features.py, it matters more than it looks.

Raw sequences are what gets stored, deliberately: the feature pipeline in features.py
will change as the project improves, and recordings of real people are too expensive to
re-collect every time it does.

Run:
    python src/record_dataset.py --label hello --samples 30

--samples is a TARGET TOTAL, not a number to add. Recording is resumable: run it again
with the same label and it tops the folder up to that many, so a session can be
interrupted without losing anything.

Keys while running:
    space   record one sample
    q       quit
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import mp_compat
import theme
from console import enable_unicode
from features import FRAMES, FRAME_FEATURES, frame_features

BASE_DIR = Path(__file__).resolve().parent.parent

# The dataset lives next to the code, not next to wherever you happened to be standing
# when you launched this. train_model.py reads from the same absolute location, and a
# default of "data" would silently split the two apart when run from another directory.
DEFAULT_DATA_DIR = BASE_DIR / "data"


def next_path(out_dir, label):
    """A path inside `out_dir` that does not already exist.

    The name used to be the label plus a whole-second timestamp, and two samples
    landing in the same second would silently overwrite each other. The countdown
    looks like it rules that out, but cv2.waitKey returns the moment a key is
    pressed rather than waiting out its timeout, so a session where someone holds
    a key down runs the whole capture in a fraction of a second.

    The damage from that is quiet, which is what makes it worth guarding: the
    recorder's own counter still goes up, so it reports 30 samples recorded while
    the folder holds 29, and the missing one is only ever noticed as an unexplained
    dip in training data.
    """
    stamp = int(time.time())
    path = out_dir / f"{label}_{stamp}.npy"
    serial = 2
    while path.exists():
        path = out_dir / f"{label}_{stamp}_{serial}.npy"
        serial += 1
    return path


def capture_sample(camera, hands, window):
    """Record one sample: a countdown, then FRAMES frames of keypoints.

    Returns the sequence, or None if the camera stopped mid-capture.
    """
    for count in (3, 2, 1):
        ok, frame = camera.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        theme.countdown(frame, count, "get ready to sign")
        cv2.imshow(window, frame)
        cv2.waitKey(600)

    sequence = np.zeros((FRAMES, FRAME_FEATURES), dtype=np.float32)
    for index in range(FRAMES):
        ok, frame = camera.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)
        sequence[index] = frame_features(result)

        theme.banner(frame, "RECORDING", theme.RED)
        theme.rule(frame, 0, 0, frame.shape[1] * (index + 1) / FRAMES, 4, theme.RED)
        cv2.imshow(window, frame)
        cv2.waitKey(1)

    return sequence


def main():
    enable_unicode()

    parser = argparse.ArgumentParser(
        description="Record sign samples. --samples is a target total; recording is "
                    "resumable and tops the folder up to that many."
    )
    parser.add_argument("--label", required=True,
                        help="sign label, e.g. hello")
    parser.add_argument("--samples", type=int, default=30,
                        help="target TOTAL samples for this label (default 30)")
    parser.add_argument("--out", default=str(DEFAULT_DATA_DIR),
                        help="dataset root folder (default: the project's data/)")
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be at least 1.")

    out_dir = Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    recorded = len(list(out_dir.glob("*.npy")))

    print(f"Label '{args.label}' -> {out_dir}")
    print(f"{recorded} samples already on disk, target is {args.samples}.")

    # Say so before opening the camera. Otherwise the window flashes open and shut with
    # no explanation, which reads as a crash.
    if recorded >= args.samples:
        raise SystemExit(
            f"Nothing to do: '{args.label}' already has {recorded} samples.\n"
            f"To add more, raise the target:  "
            f"--label {args.label} --samples {recorded + 10}"
        )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Could not open the camera. Is another app using it?")

    window = "DESTDENG - record"
    print("SPACE to record one sample, q to quit.")

    try:
        with mp_compat.new_hands() as hands:

            while recorded < args.samples:
                ok, frame = camera.read()
                if not ok:
                    print("Camera stopped.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)

                hands_seen = 0
                if result.multi_hand_landmarks:
                    hands_seen = len(result.multi_hand_landmarks)
                    for hand in result.multi_hand_landmarks:
                        mp_compat.draw(frame, hand, styled=False)

                # Whether hands are being seen is the one thing worth reporting while
                # recording. A sample captured with the hands out of frame is a
                # perfectly valid file full of zeros, and it will not announce itself
                # until training.
                if hands_seen:
                    state, colour = f"TRACKING {hands_seen}", theme.GREEN
                else:
                    state, colour = "NO HANDS IN FRAME", theme.AMBER

                bar = theme.top_bar(frame, f"RECORD  {args.label}", state, colour)

                width = frame.shape[1]
                done = recorded / args.samples
                theme.label(frame, f"{recorded} / {args.samples}", (16, bar + 30),
                            size=0.62, colour=theme.WHITE, shadow=True)
                theme.meter(frame, 16, bar + 42, int(width * 0.32), 6, done)
                theme.hint(frame, "SPACE  record a sample      q  quit",
                           y=frame.shape[0] - 16)
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if key == ord(" "):
                    sequence = capture_sample(camera, hands, window)
                    if sequence is None:
                        print("Camera stopped mid-capture; sample discarded.")
                        break

                    path = next_path(out_dir, args.label)
                    np.save(path, sequence)
                    recorded += 1
                    print(f"saved {path.name}  ({recorded}/{args.samples})")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Done. {recorded} samples for '{args.label}'.")


if __name__ == "__main__":
    main()
