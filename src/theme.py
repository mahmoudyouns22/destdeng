"""
theme.py — one visual language for every screen this project puts on a camera.

Before this file, each of the three camera screens drew its own text in its own
colour at its own hardcoded position, and they looked like three different programs.
That is a real problem and not only a cosmetic one: the person reading this screen is
usually a stranger to it — a doctor, a clerk, a police officer — who has a deaf person
waiting in front of them. They get one glance to work out where the answer appears and
whether the machine is confident. Consistency is what buys that glance.

So the palette, the panels, the pills and the countdown live here, and the screens
compose them instead of drawing raw rectangles.

THREE RULES THIS FILE ENFORCES

  NOTHING IS POSITIONED BY A MAGIC NUMBER. Every helper takes the frame and works out
  its own geometry from the real width and height. The countdown used to be drawn at a
  fixed (280, 260), which is centred only on a 640x480 camera and sits off in a corner
  on anything else.

  REFUSAL AND RECOGNITION NEVER SHARE A COLOUR. A recognised sign is white on the
  accent rule; a refusal is amber, on an amber rule, with a different icon. The reader
  must be able to tell "it understood" from "it did not" from across a desk, before
  reading a word — which matters most for the reader who cannot read the script the
  refusal is written in.

  TEXT IS DRAWN THROUGH text_render, NEVER cv2.putText, WHENEVER IT MIGHT NOT BE
  LATIN. cv2.putText has no Arabic glyphs at all. Latin chrome — key hints, FPS — may
  use cv2 directly because it is never anything else.

Colours are BGR, because that is what OpenCV takes. The RGB hex is in the comment so
the palette can be read by a human.
"""

import cv2
import numpy as np

# --- palette ----------------------------------------------------------------
# Dark, low-saturation chrome so the camera image stays the brightest thing on
# screen — the signer's hands are the content; the interface is not.

INK        = (20, 18, 16)      # #101214  panel background
INK_EDGE   = (38, 34, 30)      # #1E2226  hairline between panel and image
WHITE      = (245, 245, 247)   # #F7F5F5  recognised text
MUTED      = (150, 142, 138)   # #8A8E96  hints, secondary labels
DIM        = (95, 90, 86)      # #565A5F  inactive pill outlines

ACCENT     = (196, 205, 78)    # #4ECDC4  the project's colour: accepted, active
AMBER      = (60, 170, 250)    # #FAAA3C  refusal — "I did not understand"
GREEN      = (132, 220, 61)    # #3DDC84  live / recording indicator
RED        = (86, 86, 235)     # #EB5656  capture in progress

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Panel heights, as a fraction of frame height rather than pixels, so the layout
# survives a camera that is not 480 tall.
TOP_BAR = 0.085
OUTPUT_PANEL = 0.30


# --- primitives -------------------------------------------------------------

def overlay(frame, x0, y0, x1, y1, colour=INK, alpha=0.82):
    """Blend a filled rectangle over the frame, in place-ish.

    Returns the frame. Blending rather than filling keeps a hint of the camera image
    behind the panel, which is what stops the bottom third of the screen reading as a
    dead black bar when nothing has been recognised yet.
    """
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = int(x1), int(y1)
    if x1 <= x0 or y1 <= y0:
        return frame

    region = frame[y0:y1, x0:x1]
    tint = np.full(region.shape, colour, dtype=np.uint8)
    frame[y0:y1, x0:x1] = cv2.addWeighted(tint, alpha, region, 1 - alpha, 0)
    return frame


def rule(frame, x, y, width, height=3, colour=ACCENT):
    """A short accent bar. Carries state by colour — see the note in the docstring."""
    cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)),
                  colour, -1)
    return frame


def label(frame, text, org, size=0.5, colour=MUTED, weight=1, shadow=False):
    """Latin chrome only. Anything that might not be Latin goes through text_render.

    `shadow` draws the string once in near-black underneath. Text inside a panel does
    not need it; text laid directly over the camera image does, because the image is
    whatever the room happens to look like — a grey hint over a bright wall or a lit
    face is not dimly legible, it is gone.
    """
    org = (int(org[0]), int(org[1]))
    if shadow:
        cv2.putText(frame, text, org, FONT, size, (12, 12, 12), weight + 2,
                    cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, size, colour, weight, cv2.LINE_AA)
    return frame


def pill(frame, text, org, fg=WHITE, border=DIM, fill=None, size=0.46, pad=9):
    """A bordered chip. Returns the x just past its right edge, so chips can be
    laid out in a row without the caller measuring anything itself."""
    (tw, th), _ = cv2.getTextSize(text, FONT, size, 1)
    x, y = int(org[0]), int(org[1])
    x1, y1 = x + tw + pad * 2, y + th + pad

    if fill is not None:
        cv2.rectangle(frame, (x, y), (x1, y1), fill, -1)
    cv2.rectangle(frame, (x, y), (x1, y1), border, 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x + pad, y1 - pad // 2), FONT, size, fg, 1, cv2.LINE_AA)
    return x1


def dot(frame, centre, colour, radius=4):
    cv2.circle(frame, (int(centre[0]), int(centre[1])), radius, colour, -1, cv2.LINE_AA)
    return frame


def meter(frame, x, y, width, height, fraction, colour=ACCENT, track=(52, 48, 44)):
    """A confidence bar.

    A percentage is a number the reader has to interpret; a bar is a length they can
    see. Both are shown — the bar for the glance, the number for the record.
    """
    fraction = min(max(float(fraction), 0.0), 1.0)
    x, y, width, height = int(x), int(y), int(width), int(height)
    cv2.rectangle(frame, (x, y), (x + width, y + height), track, -1)
    filled = int(width * fraction)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), colour, -1)
    return frame


# --- composed screen furniture ----------------------------------------------

def top_bar(frame, title="DESTDENG", state=None, state_colour=GREEN):
    """The strip along the top: the product mark, and what the system is doing.

    Returns the y coordinate just below the bar, so callers never guess where the
    camera image starts.
    """
    height, width = frame.shape[:2]
    bar = max(34, int(height * TOP_BAR))

    overlay(frame, 0, 0, width, bar, INK, 0.72)
    cv2.line(frame, (0, bar), (width, bar), INK_EDGE, 1)

    baseline = bar // 2 + 6
    dot(frame, (18, bar // 2), ACCENT, 4)
    cv2.putText(frame, title, (32, baseline), FONT, 0.58, WHITE, 1, cv2.LINE_AA)

    if state:
        (tw, _), _ = cv2.getTextSize(state, FONT, 0.46, 1)
        x = width - tw - 34
        dot(frame, (x - 12, bar // 2), state_colour, 4)
        cv2.putText(frame, state, (x, baseline), FONT, 0.46, state_colour, 1,
                    cv2.LINE_AA)

    return bar


def language_row(frame, y, languages, active, x=14):
    """The output-language chips: which language the reader is being shown.

    Rendered as three chips with the live one filled, rather than as the old
    'Output: Arabic [1/2/3 to change]' line. The reader is often not the person
    holding the keyboard, and a filled chip says which language is on screen
    without asking them to read a sentence about it first.
    """
    for index, code in enumerate(languages, start=1):
        chosen = code == active
        x = pill(
            frame, f"{index} {code.upper()}", (x, y),
            fg=INK if chosen else MUTED,
            border=ACCENT if chosen else DIM,
            fill=ACCENT if chosen else None,
        ) + 8
    return x


def countdown(frame, number, subtitle=None):
    """The 3-2-1 before a capture, centred on the frame it is actually given.

    Centre comes from the frame, not from a constant. The previous fixed position was
    correct for one camera resolution and wrong for every other one.
    """
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2

    overlay(frame, 0, 0, width, height, INK, 0.35)

    radius = int(min(width, height) * 0.13)
    cv2.circle(frame, (cx, cy), radius, ACCENT, 2, cv2.LINE_AA)

    text = str(number)
    scale = radius / 26.0
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 4)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2), FONT, scale, WHITE, 4,
                cv2.LINE_AA)

    if subtitle:
        (sw, _), _ = cv2.getTextSize(subtitle, FONT, 0.55, 1)
        cv2.putText(frame, subtitle, (cx - sw // 2, cy + radius + 34), FONT, 0.55,
                    MUTED, 1, cv2.LINE_AA)
    return frame


def banner(frame, text, colour=RED):
    """A wide status strip across the middle — used while a sign is being read.

    Placed centrally rather than in a corner because during a capture the signer is
    looking at the camera, not hunting the interface.
    """
    height, width = frame.shape[:2]
    band = int(height * 0.11)
    y0 = (height - band) // 2

    overlay(frame, 0, y0, width, y0 + band, INK, 0.62)
    rule(frame, 0, y0, width, 3, colour)

    (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    cv2.putText(frame, text, ((width - tw) // 2, y0 + (band + th) // 2), FONT, 0.8,
                colour, 2, cv2.LINE_AA)
    return frame


def output_panel_top(frame):
    """Where the output panel starts.

    Both the panel and the key legend need this number, and when each worked it out
    for itself they disagreed: the legend sat at the bottom of the frame, which is
    inside the panel, printed across the line naming the recognised sign.
    """
    height = frame.shape[0]
    return height - max(120, int(height * OUTPUT_PANEL))


def hint(frame, text, y=None):
    """The key legend, in the quietest colour on the palette.

    Sits just above the output panel by default, not at the bottom of the frame. The
    panel is for the reader; the keys are for whoever is driving. Keeping the two
    apart is what stops the legend printing over the result.
    """
    if y is None:
        y = output_panel_top(frame) - 12
    return label(frame, text, (16, y), size=0.45, colour=MUTED, shadow=True)
