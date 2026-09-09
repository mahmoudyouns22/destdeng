"""
text_render.py — draw Arabic, Kurdish and English text onto an OpenCV frame.

OpenCV's cv2.putText cannot draw Arabic script at all: it has no glyphs for it, and it
would not join the letters even if it did. Since showing the text in the reader's own
language is the whole point of this project, we render text with Pillow instead, and
reshape Arabic script so the letters connect and run right to left.

Three things this file has to get right, none of them obvious:

  KURDISH IS NOT ARABIC, AND ITS SHAPING MODE IS A TRAP. Sorani uses letters Arabic
  does not, and arabic-reshaper has a Kurdish mode for them. Switching it on and
  trusting it is wrong: four of those letters — ە ڵ ێ and their forms — have no
  Unicode presentation forms at all, so the library emits PRIVATE USE codepoints
  (U+E000, U+E004, U+E006) for them. Those slots are empty in every ordinary font, so
  Tahoma, Segoe UI and Arial all draw them as hollow boxes. سڵاو came out fine and
  تێنەگەیشتم came out as ت□نە□گە□یشتم — in the message shown when the system has
  failed, to a reader who cannot check it against anything.

  The mode is only correct with a font that fills those private-use slots. So the
  choice is made by testing, not by assumption: we shape a probe with Kurdish mode
  and check the chosen font can actually draw the result. If it cannot, we fall back
  to the default mode, which uses real Unicode presentation forms, needs no special
  font, and leaves only ڵ unjoined — legible Kurdish instead of boxes.

  THE PROBE IS THE REAL VOCABULARY, SHAPED. Checking the raw letters proves nothing:
  every candidate font contains all 42 of them and still fails, because what gets
  drawn is the reshaper's output, not the input. So the coverage test runs over the
  shaped form of every string this program can ever display.

  RIGHT-TO-LEFT TEXT IS RIGHT-ALIGNED. Drawing Arabic from a fixed left margin leaves
  it stranded mid-frame with a gap exactly where the reader's eye starts. RTL callers
  pass the right edge instead, and we measure the string to place it.

  FAILURES ARE NOT SILENT. If shaping breaks, Arabic still renders — as disconnected
  letters in the wrong order, which reads as broken language rather than broken
  software. So a shaping failure is recorded and reported, never swallowed.

Everything degrades rather than crashes: with no Pillow, no shaping libraries or no
suitable font, the caller still gets a frame back with the English text drawn on it.
"""

import os

import cv2
import numpy as np

import vocabulary

# --- optional dependencies -------------------------------------------------

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    SHAPING_OK = True
except ImportError:
    SHAPING_OK = False


# Recorded rather than swallowed, and surfaced by startup_report().
_problems = []


def _note(problem):
    if problem not in _problems:
        _problems.append(problem)


# --- what actually has to be drawable --------------------------------------

def _probe_text():
    """Every non-Latin string this program can ever put on screen.

    The font and shaping checks below run against this rather than against a
    hand-written sample of letters. A sample is what hid the private-use problem for
    as long as it was hidden: every candidate font contains all 42 raw letters the
    vocabulary uses, and three of them still drew boxes, because what reaches the
    font is the reshaper's output and not its input.
    """
    parts = []
    for table in (vocabulary.VOCABULARY, vocabulary.MESSAGES):
        for entry in table.values():
            for language in ("ar", "ku"):
                parts.append(entry[language])
    return " ".join(parts)


# --- font discovery --------------------------------------------------------

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahoma.ttf",         # Windows — widest Arabic-script coverage
    r"C:\Windows\Fonts\segoeui.ttf",        # Windows — covers the Kurdish letters
    r"C:\Windows\Fonts\arial.ttf",          # Windows — Arabic, but patchier for Sorani
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",            # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",         # Linux
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]

# A codepoint no font legitimately contains, used as the reference for "this font has
# nothing for that character". Rendered pixels are compared rather than bounding
# boxes: a .notdef box and a real glyph can share a bounding box, which is precisely
# how the private-use problem went unnoticed.
NOTHING = "\U0010fffd"


def _ink(font, character):
    """What this font actually draws for one character, or None if it cannot."""
    try:
        image = Image.new("L", (72, 72), 0)
        ImageDraw.Draw(image).text((8, 8), character, font=font, fill=255)
        return np.array(image).tobytes()
    except Exception:
        return None


def _glyphless(font, text):
    """The characters of `text` this font has nothing to draw."""
    reference = _ink(font, NOTHING)
    missing = []
    for character in dict.fromkeys(text):          # unique, order preserved
        if character.isspace() or ord(character) < 0x80:
            continue
        drawn = _ink(font, character)
        if drawn is None or drawn == reference:
            missing.append(character)
    return missing


def _reshape_only(reshaper, text):
    """Shape without reordering — enough to see which glyphs a font gets asked for."""
    if reshaper is None:
        return arabic_reshaper.reshape(text)
    return reshaper.reshape(text)


def _kurdish_reshaper():
    if not SHAPING_OK:
        return None
    try:
        return arabic_reshaper.ArabicReshaper({"language": "Kurdish"})
    except Exception:
        _note("arabic-reshaper has no Kurdish mode; Sorani letters may not join")
        return None


def _choose_font_and_shaping():
    """Pick the font and the shaping mode together, by testing what each can draw.

    They cannot be chosen separately. Kurdish mode is correct only with a font that
    fills the private-use slots it emits for ە ڵ ێ; with any ordinary font it
    produces boxes, and the failure lands hardest on the refusal message, which is
    the one string the reader has no way to check against anything. So each candidate
    font is offered each mode, and the first pair that can actually be drawn wins.

    Returns (font path, reshaper, mode name).
    """
    if not PIL_OK:
        return None, None, "none"

    probe = _probe_text()
    kurdish = _kurdish_reshaper()

    modes = []
    if kurdish is not None:
        modes.append(("kurdish", kurdish))
    if SHAPING_OK:
        modes.append(("default", None))

    faces = []
    for path in FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            faces.append((path, ImageFont.truetype(path, 24)))
        except Exception:
            continue

    # Mode is the outer loop, font the inner one, and that order is the point. Kurdish
    # mode joins ڵ ڕ ۆ ێ properly and the default mode cannot — Unicode has no
    # presentation forms for them — so a later font that unlocks Kurdish mode is worth
    # more than the first font in the list running in the default one. Installing a
    # Kurdish font that fills those private-use slots and adding it to FONT_CANDIDATES
    # is therefore all it takes to upgrade the typography; nothing else has to change.
    for name, reshaper in modes:
        try:
            shaped = _reshape_only(reshaper, probe)
        except Exception:
            continue
        for path, face in faces:
            if not _glyphless(face, shaped):
                if name == "default" and kurdish is not None:
                    _note("no candidate font has glyphs for the private-use "
                          "codepoints Kurdish shaping emits, so standard shaping is "
                          "in use with {}; ڵ and ڕ will not join"
                          .format(os.path.basename(path)))
                return path, reshaper, name

    # Unshaped is a last resort worth having: every letter is present, they simply
    # will not join. Disconnected Kurdish still beats a row of boxes.
    for path, face in faces:
        if not _glyphless(face, probe):
            _note("{} cannot draw the output of either shaping mode; Kurdish and "
                  "Arabic will show unjoined".format(os.path.basename(path)))
            return path, None, "unshaped"

    fallback = faces[0][0] if faces else None

    if fallback is not None:
        _note("no installed font covers the vocabulary; using {}, so some Kurdish "
              "will show as boxes".format(os.path.basename(fallback)))
    return fallback, None, "default" if SHAPING_OK else "none"


FONT_PATH, _RESHAPER, SHAPING_MODE = _choose_font_and_shaping()


def shape(text):
    """Reshape Arabic script so letters join, and reorder it for right-to-left display."""
    if not SHAPING_OK or SHAPING_MODE in ("unshaped", "none"):
        return text
    try:
        return get_display(_reshape_only(_RESHAPER, text))
    except Exception as error:
        _note("text shaping failed ({}); Arabic and Kurdish will render "
              "unjoined and reversed".format(type(error).__name__))
        return text

# Building a font face costs milliseconds. The recognition loop draws text on every
# frame, so building it per frame would show up directly in the frame rate.
_font_cache = {}


def _font(size):
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
        except Exception as error:
            _note("could not load {} ({})".format(FONT_PATH, type(error).__name__))
            _font_cache[size] = None
    return _font_cache[size]


def can_render_rtl():
    """True if we can actually draw joined right-to-left script.

    `unshaped` is excluded deliberately. The letters do appear in that mode, so it is
    tempting to count it as working, but they stand apart and run the wrong way — the
    reader sees their own language spelled out backwards in pieces, which is worse
    than being told plainly that it cannot be displayed.
    """
    return (PIL_OK and SHAPING_OK and FONT_PATH is not None
            and SHAPING_MODE in ("kurdish", "default"))


# --- drawing ---------------------------------------------------------------

def _putText_fallback(frame, text, position, size, colour, rtl=False):
    """Last resort when Pillow or a usable font is missing.

    `rtl` still has to be honoured here even though this path cannot shape Arabic.
    An RTL caller passes the RIGHT edge as x, so drawing left-to-right from it puts
    the whole string off the side of the frame — the reader gets a blank panel rather
    than degraded text, which looks like a crash instead of a missing font.
    """
    scale = size / 40.0
    x, y = int(position[0]), int(position[1])
    if rtl:
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = max(0, x - width)
    cv2.putText(frame, text, (x, y + size),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)
    return frame


def text_width(text, size, rtl=False):
    """How wide `text` will actually be when drawn at `size`, in pixels.

    Measured after shaping, because shaping changes the width: Arabic letters written
    in their joined forms are narrower than the same letters standing alone, and
    measuring the unshaped string overestimates by enough to shrink text that would
    have fitted.
    """
    display = shape(text) if rtl else text

    if PIL_OK and FONT_PATH:
        font = _font(size)
        if font is not None:
            try:
                return float(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(
                    display, font=font))
            except Exception:
                box = font.getbbox(display)
                return float(box[2] - box[0])

    (width, _), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX,
                                    size / 40.0, 2)
    return float(width)


def fit_size(text, max_width, size, minimum=18, rtl=False, latin=None):
    """The largest size at or below `size` at which `text` fits inside `max_width`.

    `latin` mirrors draw_text: measure whichever string is actually going to be
    drawn, or the fit is computed for one string and applied to another.

    Scaling by the measured ratio lands within a point or two immediately, so the
    loop below is a correction rather than a search — it matters because this runs on
    every frame of a real-time loop, on the CPU the project promises to run on.
    """
    if latin is not None and not (PIL_OK and FONT_PATH):
        text, rtl = latin, False

    if not text or max_width <= 0:
        return size

    width = text_width(text, size, rtl)
    if width <= max_width:
        return size

    size = max(minimum, int(size * max_width / width))
    while size > minimum and text_width(text, size, rtl) > max_width:
        size -= 1
    return size


def draw_text(frame, text, position, size=44, colour=(255, 255, 255), rtl=False,
              latin=None):
    """Draw `text` onto an OpenCV BGR frame.

    `position` is (x, y) of the top-left corner — except when `rtl` is true, where x is
    the right edge the text should end at, because that is where a right-to-left
    reader's eye starts.

    `latin` is what to draw instead when this machine cannot draw Arabic script at
    all. It matters more than it sounds. cv2.putText, the only thing left when Pillow
    is missing, has no glyphs outside ASCII — it does not render Arabic badly, it
    renders every character of it as a literal question mark. Without `latin` the
    reader is handed ?????? and told nothing; with it they get Supas, which is their
    own language in the wrong alphabet and still readable. The caller knows which
    romanisation belongs to the string; this function cannot invent one.
    """
    if not (PIL_OK and FONT_PATH):
        return _putText_fallback(frame, latin or text, position, size, colour, rtl)

    font = _font(size)
    if font is None:
        return _putText_fallback(frame, latin or text, position, size, colour, rtl)

    display = shape(text) if rtl else text

    # OpenCV is BGR, Pillow is RGB.
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    x, y = position
    if rtl:
        try:
            width = draw.textlength(display, font=font)
        except AttributeError:            # Pillow < 8
            box = font.getbbox(display)
            width = box[2] - box[0]
        x = max(0, int(x - width))

    draw.text((x, y), display, font=font, fill=(colour[2], colour[1], colour[0]))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def startup_report():
    """One or more lines about what this machine can and cannot display."""
    missing = []
    if not PIL_OK:
        missing.append("Pillow")
    if not SHAPING_OK:
        missing.append("arabic-reshaper, python-bidi")
    if FONT_PATH is None:
        missing.append("a system font with Arabic glyphs")

    if missing:
        lines = ["Text rendering: LIMITED - Arabic and Kurdish may not display "
                 "correctly. Missing: {}".format(", ".join(missing))]
    else:
        # The shaping mode is reported, not just the font. Which mode is in use is
        # the difference between joined Kurdish and a line of boxes, and it is chosen
        # at import from what this machine's fonts can draw — so it is a property of
        # the machine the program is running on, and belongs in the startup line.
        lines = ["Text rendering: full (font: {}, shaping: {})".format(
            os.path.basename(FONT_PATH), SHAPING_MODE)]

    lines.extend("  warning: {}".format(problem) for problem in _problems)
    return "\n".join(lines)
