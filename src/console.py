"""
console.py — make the terminal able to print Arabic and Kurdish.

On Windows, Python's stdout defaults to the legacy code page (cp1252 on most machines
here). Printing a single Arabic or Kurdish character to it raises UnicodeEncodeError
and takes the whole program down — which for this project means the vocabulary
listing crashes on exactly the languages the project exists to display.

Calling enable_unicode() once at the top of an entry point fixes that. Characters the
console font cannot draw come out as replacement marks instead of killing the process,
which is the right trade: a missing glyph is a display problem, a crash is not.
"""

import os
import sys


def enable_unicode():
    """Switch stdout/stderr to UTF-8, replacing anything the console cannot draw.

    Returns True if stdout is now UTF-8, which the menu uses to decide whether it can
    draw box-drawing characters or has to fall back to ASCII. A console that turns
    ╭──╮ into ??? looks more broken than one that was only ever going to use +--+.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Already-detached or redirected stream. Nothing to do, and nothing broken.
            pass

    return (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") == "utf8"


def enable_colour():
    """Turn on ANSI colour, and report whether it is actually usable.

    Three things have to hold, and each is checked rather than assumed:

      The output is a terminal. Piped into a file or a pipe, escape codes are not
      colour, they are litter in the middle of the text.

      NO_COLOR is unset. It is the standard way to ask a program not to colourise,
      and honouring it costs one line.

      Windows has virtual-terminal processing switched on. Consoles here do not
      interpret ANSI codes until a program asks them to, and one that never asks
      prints the raw escape sequences instead of colouring anything.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if not getattr(sys.stdout, "isatty", lambda: False)():
        return False

    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            if not kernel32.SetConsoleMode(
                    handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING):
                return False
        except Exception:
            return False

    return True
