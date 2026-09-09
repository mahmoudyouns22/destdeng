"""
recognise.py — DESTDENG, live.

This is the product. Point a webcam at someone signing, press SPACE, and the recognised
sign appears as text in whichever language the reader picked.

Run:
    python src/recognise.py

Keys:
    SPACE   capture and recognise one sign
    1       output in English
    2       output in Arabic
    3       output in Kurdish
    q       quit

The design rule that matters most is enforced in src/confidence.py: when the input is
unlike anything trained on, or the classifier is not sure which sign it is, the system
does NOT show its best guess. It says it did not understand and asks for the sign
again. A wrong translation in a hospital is worse than no translation, so the metric
this project is judged by is not accuracy — it is how rarely it asserts something false
with confidence.

That refusal is worded as a system message, never as a sign. "Please repeat" is itself
a sign in the vocabulary, and a reader must never have to guess whether the deaf person
asked them to repeat or the machine failed. See the note above MESSAGES in
src/vocabulary.py.

THE SAME RULE, CARRIED INTO THE INTERFACE
    A refusal is not only worded differently, it is coloured differently: accepted
    signs come up white on the accent rule, refusals amber on an amber rule, with the
    confidence meter withheld rather than shown low. The reader gets one glance at
    this screen across a desk, and often cannot read the script the message is written
    in. "It understood" and "it did not" have to survive that glance. The palette and
    the panels live in src/theme.py so all three camera screens agree on them.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import confidence
import features as feat
import mp_compat
import text_render
import theme
from console import enable_unicode
from vocabulary import LANGUAGES, is_rtl, message, to_text

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sign_classifier.pkl"

RETRAIN = "Retrain it:  python src/train_model.py"


def load_model():
    """Load the trained bundle, refusing anything built on an older feature layout."""
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No trained model at {MODEL_PATH}\n\n"
            "Record samples and train first:\n"
            "    python src/record_dataset.py --label hello --samples 30\n"
            "    python src/record_dataset.py --label thank_you --samples 30\n"
            "    python src/train_model.py"
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "novelty" not in bundle:
        raise SystemExit(
            f"This model file predates the novelty check.\n{RETRAIN}"
        )

    version = bundle.get("feature_version")
    if version != feat.FEATURE_VERSION:
        # Silently scoring new features against an old model would produce confident
        # nonsense, which is the one failure mode this project cannot tolerate.
        raise SystemExit(
            f"This model was trained on feature layout v{version}, but the code now "
            f"produces v{feat.FEATURE_VERSION}.\n{RETRAIN}"
        )

    return bundle["model"], bundle["novelty"]


def capture_sequence(camera, hands, window="DESTDENG"):
    """Record one sequence of keypoints, with a countdown so the signer can start."""
    for count in (3, 2, 1):
        ok, frame = camera.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        theme.countdown(frame, count, "get ready to sign")
        cv2.imshow(window, frame)
        cv2.waitKey(600)

    sequence = np.zeros((feat.FRAMES, feat.FRAME_FEATURES), dtype=np.float32)
    for index in range(feat.FRAMES):
        ok, frame = camera.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        sequence[index] = feat.frame_features(hands.process(rgb))

        theme.banner(frame, "READING SIGN", theme.RED)
        # A progress rule along the very top: the signer needs to know how much of
        # the window is left, or they finish the sign into a camera that stopped
        # recording half a second ago.
        theme.rule(frame, 0, 0, frame.shape[1] * (index + 1) / feat.FRAMES, 4,
                   theme.RED)
        cv2.imshow(window, frame)
        cv2.waitKey(1)

    return sequence


def predict(model, novelty, sequence):
    """Recognise one captured sequence, or refuse. See src/confidence.py."""
    vector = feat.sequence_features(sequence).reshape(1, -1)
    return confidence.decide(model, novelty, vector)


class Reading:
    """One completed attempt at reading a sign, held as a label rather than as text.

    Keeping the label — not the rendered string — is what lets the reader switch
    output language after the fact and see the same result in the new language.
    `key` is a vocabulary label when accepted and a MESSAGES key when refused; the
    two namespaces are deliberately disjoint (see src/vocabulary.py), so `accepted`
    is all that is needed to know which table to read it from.
    """

    def __init__(self, accepted, key, detail, probability):
        self.accepted = accepted
        self.key = key
        self.detail = detail
        self.probability = probability

    def text(self, language):
        return (to_text(self.key, language) if self.accepted
                else message(self.key, language))

    def latin(self, language):
        """The same result in the Latin alphabet, for a machine that cannot draw
        Arabic script at all. Kurdish has a romanisation in the vocabulary; Arabic
        does not, so an Arabic reader falls back to English rather than to a
        transliteration nobody uses."""
        return self.text("ku_latin" if language == "ku" else "en")


def draw_output_panel(frame, last, language):
    """The bottom third of the screen — the only part the reader actually reads.

    Returns a frame, because drawing non-Latin text round-trips through Pillow and
    hands back a new array rather than editing this one in place.
    """
    height, width = frame.shape[:2]
    top = theme.output_panel_top(frame)
    panel = height - top

    theme.overlay(frame, 0, top, width, height, theme.INK, 0.88)
    accepted = last.accepted if last else True
    accent = theme.ACCENT if accepted else theme.AMBER
    theme.rule(frame, 0, top, width, 3, accent)

    if last is None:
        # Nothing read yet. Say what to do, in the quietest colour available, so an
        # idle screen never looks like a screen that has failed.
        theme.label(frame, "Press SPACE to read a sign", (26, top + panel // 2 + 6),
                    size=0.62, colour=theme.DIM)
        return frame

    text = last.text(language)
    rtl = is_rtl(language)
    # For right-to-left text the x is the right edge, because that is where the
    # reader starts.
    margin = 26
    x = width - margin if rtl else margin

    # Shrink to fit rather than run off the edge. A recognised sign is one short word
    # and always fits; a refusal is a whole sentence, and the Kurdish one is the
    # longest string this panel ever has to show. Overflowing would cut the end off
    # an RTL line at the LEFT edge — the end of the sentence — so the reader sees a
    # message that looks complete and is not.
    latin = last.latin(language)
    size = text_render.fit_size(text, width - margin * 2,
                                max(34, int(panel * 0.40)), 20, rtl=rtl, latin=latin)
    frame = text_render.draw_text(
        frame, text, (x, top + int(panel * 0.16)),
        size=size, colour=theme.WHITE if accepted else theme.AMBER, rtl=rtl,
        latin=latin,
    )

    # Meta row: what it decided and how sure it was. The meter is drawn only for an
    # accepted sign — showing a low bar next to a refusal invites the reader to treat
    # the refused guess as a weak answer, when it is not an answer at all.
    meta_y = height - 20
    theme.label(frame, last.detail, (26, meta_y), size=0.46,
                colour=theme.MUTED if accepted else theme.AMBER)

    if accepted and last.probability is not None:
        probability = last.probability
        bar_width = int(width * 0.22)
        bar_x = width - bar_width - 26
        theme.meter(frame, bar_x, meta_y - 9, bar_width, 7, probability, accent)
        theme.label(frame, f"{probability:.0%}", (bar_x - 44, meta_y), size=0.46,
                    colour=theme.MUTED)

    return frame


def main():
    enable_unicode()

    model, novelty = load_model()
    print("Model loaded.")
    print(text_render.startup_report())
    print("\nSPACE = read a sign   |   1 English  2 Arabic  3 Kurdish   |   q = quit\n")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Could not open the camera. Is another app using it?")

    window = "DESTDENG"
    language = "en"

    # What was last read, kept as a LABEL rather than as rendered text. The reader can
    # change the output language after a sign has been read — at a hospital desk that
    # is the normal case, not an edge case, because the person who presses the key is
    # often not the person who has to read the word. Storing the rendered string meant
    # the panel kept showing the old language until the next sign was made.
    last = None

    try:
        with mp_compat.new_hands() as hands:

            while True:
                ok, frame = camera.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                tracked = hands.process(rgb)

                hands_seen = 0
                if tracked.multi_hand_landmarks:
                    hands_seen = len(tracked.multi_hand_landmarks)
                    for hand in tracked.multi_hand_landmarks:
                        mp_compat.draw(frame, hand)

                # State in the top bar, from the tracker rather than from a guess:
                # "no hands in frame" is the most common reason a capture fails, and
                # it is the one thing the signer can fix before pressing SPACE.
                if hands_seen:
                    state, state_colour = f"TRACKING {hands_seen}", theme.GREEN
                else:
                    state, state_colour = "NO HANDS IN FRAME", theme.AMBER

                bar = theme.top_bar(frame, "DESTDENG", state, state_colour)
                theme.language_row(frame, bar + 12, list(LANGUAGES), language)

                frame = draw_output_panel(frame, last, language)
                theme.hint(frame, "SPACE  read a sign      1 2 3  language      q  quit")

                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("1"):
                    language = "en"
                if key == ord("2"):
                    language = "ar"
                if key == ord("3"):
                    language = "ku"

                if key == ord(" "):
                    sequence = capture_sequence(camera, hands, window)
                    if sequence is None:
                        print("Camera stopped.")
                        break

                    decision = predict(model, novelty, sequence)

                    if decision.accepted:
                        last = Reading(True, decision.label, decision.label,
                                       decision.probability)
                        print(f"  {decision.label}  ({decision.probability:.0%})")
                    else:
                        # The rule: never assert something false with confidence. The
                        # wording is a system message, so it can never be mistaken for
                        # the "repeat" sign the signer might genuinely have made.
                        key_name = ("no_hands" if decision.reason == "no hands detected"
                                    else "not_understood")
                        last = Reading(False, key_name,
                                       f"not recognised - {decision.reason}", None)
                        print(f"  refused: {decision.reason}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
