"""
mp_compat.py — one place that gets MediaPipe's hand tracking, with a clear error.

Two things go wrong here often enough to be worth catching properly, because both
surface far from their cause:

  WRONG PYTHON. MediaPipe 0.10.14 publishes wheels for Python 3.8 - 3.12 only. On a
  newer interpreter `pip install -r requirements.txt` fails with "no matching
  distribution", which reads like a broken requirements file rather than a Python
  that is too new.

  WRONG MEDIAPIPE. Newer MediaPipe releases dropped the classic `mediapipe.solutions`
  API this project uses. When pip resolves to one of those, the failure is an
  AttributeError deep inside unrelated code.

Both turn into a message here that says what to do about it.
"""

import sys

SUPPORTED_PYTHON = ((3, 8), (3, 12))
REQUIRED_MEDIAPIPE = "0.10.14"


def _python_note():
    low, high = SUPPORTED_PYTHON
    running = "{}.{}.{}".format(*sys.version_info[:3])
    note = (
        f"  your Python : {running}\n"
        f"  supported   : {low[0]}.{low[1]} - {high[0]}.{high[1]}\n"
    )
    if not (low <= sys.version_info[:2] <= high):
        note += (
            "\nYour Python is outside that range, which is almost certainly the cause.\n"
            "Create the virtual environment with a supported one, for example:\n\n"
            f"    py -{high[0]}.{high[1]} -m venv .venv\n"
            "    .venv\\Scripts\\activate\n"
            "    pip install -r requirements.txt\n"
        )
    return note


try:
    import mediapipe as mp
except ImportError:
    sys.exit(
        "MediaPipe is not installed, so hand tracking cannot start.\n\n"
        f"    pip install \"mediapipe=={REQUIRED_MEDIAPIPE}\"\n\n"
        + _python_note()
    )


if not hasattr(mp, "solutions"):
    sys.exit(
        "This MediaPipe version does not provide the hand tracking API this project "
        "uses.\n\n"
        f"  installed : mediapipe {getattr(mp, '__version__', 'unknown')}\n"
        f"  required  : mediapipe {REQUIRED_MEDIAPIPE}  "
        "(the last line that ships mediapipe.solutions)\n\n"
        "Fix it:\n\n"
        f"    pip install \"mediapipe=={REQUIRED_MEDIAPIPE}\"\n\n"
        + _python_note()
    )


hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
styles = mp.solutions.drawing_styles

HAND_CONNECTIONS = hands.HAND_CONNECTIONS


def new_hands(max_num_hands=2, detection=0.6, tracking=0.6):
    """A Hands tracker configured for real-time use on a CPU."""
    return hands.Hands(
        model_complexity=0,           # lightest model — keeps it real-time on CPU
        max_num_hands=max_num_hands,  # sign language is often two-handed
        min_detection_confidence=detection,
        min_tracking_confidence=tracking,
    )


def draw(frame, hand_landmarks, styled=True):
    """Draw one hand's skeleton onto a frame."""
    if styled:
        drawing.draw_landmarks(
            frame, hand_landmarks, HAND_CONNECTIONS,
            styles.get_default_hand_landmarks_style(),
            styles.get_default_hand_connections_style(),
        )
    else:
        drawing.draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS)
