"""
landmarks.py — live hand landmark extraction from a webcam.

This is the first stage of the DESTDENG pipeline: instead of feeding raw pixels to a
classifier, we extract 21 hand keypoints per hand, per frame. Everything downstream
works on those keypoints.

Why keypoints and not pixels:
  - Runs in real time on an ordinary CPU. No GPU needed.
  - Ignores skin tone, clothing, lighting and background, none of which carry meaning
    in sign language.
  - Needs far less training data, which matters a lot when the dataset for Kurdish
    Sign Language has to be recorded from scratch.

Run as a script it is the camera test: it draws the skeleton and reports the frame
rate, plus which hand MediaPipe assigned to which feature slot, since that assignment
is what keeps two-handed signs consistent (see src/features.py).

The maths itself lives in src/features.py, which imports neither OpenCV nor MediaPipe
so it stays testable without a camera.

Run:
    python src/landmarks.py
Quit:
    press q
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

import mp_compat
import theme
from console import enable_unicode
from features import HAND_FEATURES, frame_features, normalise  # noqa: F401  (re-exported)


def main():
    enable_unicode()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Could not open the camera. Is another app using it?")

    last = time.time()
    fps = 0.0

    try:
        with mp_compat.new_hands() as hands:

            while True:
                ok, frame = camera.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)                      # mirror, feels natural
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # MediaPipe expects RGB
                rgb.flags.writeable = False
                result = hands.process(rgb)

                hand_count = 0
                if result.multi_hand_landmarks:
                    hand_count = len(result.multi_hand_landmarks)
                    for hand in result.multi_hand_landmarks:
                        mp_compat.draw(frame, hand)

                # The exact vector the recorder and the recogniser would store for this
                # frame. Showing which slots are filled makes a handedness problem
                # visible here rather than as unexplained accuracy loss after training.
                features = frame_features(result)
                slots = "".join(
                    "#" if features[i * HAND_FEATURES:(i + 1) * HAND_FEATURES].any()
                    else "-"
                    for i in (0, 1)
                )

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
                last = now

                # Below 15 fps the tracker starts missing the fast part of a sign, so
                # the frame rate is not a statistic here — it is a pass/fail reading,
                # and it is coloured like one. SETUP.md documents 15 as the floor.
                healthy = fps >= 15.0
                bar = theme.top_bar(
                    frame, "CAMERA TEST", f"{fps:4.1f} FPS",
                    theme.GREEN if healthy else theme.AMBER)

                theme.label(frame, f"hands {hand_count}    slots {slots}",
                            (16, bar + 30), size=0.6,
                            colour=theme.WHITE if hand_count else theme.DIM,
                            shadow=True)

                # What the slots mean, spelled out. '##' or '#-' is meaningless to
                # anyone who has not read features.py, and this screen exists to tell
                # a person whether their camera setup works.
                # ASCII only: these go through cv2.putText, which has no glyph for
                # anything outside it and silently substitutes a question mark.
                reading = {"##": "both hands assigned",
                           "#-": "one hand - left slot",
                           "-#": "one hand - right slot",
                           "--": "no hands in frame"}[slots]
                theme.label(frame, reading, (16, bar + 54), size=0.48,
                            colour=theme.MUTED, shadow=True)

                if not healthy:
                    theme.label(frame, "below 15 FPS - close other apps",
                                (16, bar + 78), size=0.48, colour=theme.AMBER,
                                shadow=True)

                theme.hint(frame, "q  quit", y=frame.shape[0] - 16)

                cv2.imshow("DESTDENG - landmark extraction", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
