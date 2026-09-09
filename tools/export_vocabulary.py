"""
export_vocabulary.py — generate web/js/vocabulary.js from src/vocabulary.py.

src/vocabulary.py says it is "the only place that maps a label to words", and that has
to stay true now that a browser shows those words too. Two hand-maintained copies of a
vocabulary drift, and the drift is invisible: nothing crashes, a sign simply reads
correctly on the desktop and wrongly, or not at all, on the web — in a language the
person maintaining it may not read.

So the JavaScript file is generated, never edited. Run this after changing the
vocabulary:

    python tools/export_vocabulary.py

tools/check_parity.py fails if the generated file is out of date, so a forgotten run
is caught rather than shipped.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import vocabulary  # noqa: E402
from console import enable_unicode  # noqa: E402

TARGET = ROOT / "web" / "js" / "vocabulary.js"

HEADER = """/**
 * vocabulary.js — GENERATED FILE. Do not edit.
 *
 * Written by tools/export_vocabulary.py from src/vocabulary.py, which remains the
 * only place that maps a label to words. Edit that, then re-run:
 *
 *     python tools/export_vocabulary.py
 *
 * The reasoning behind the Kurdish column, and behind MESSAGES being separate from
 * VOCABULARY, is in src/vocabulary.py and is not duplicated here. The short version
 * of the second one matters enough to repeat: "Please repeat" is a sign a person can
 * genuinely make, so the message shown when the system fails must never be able to
 * read as something the signer said.
 */
"""

BODY = """
export const VOCABULARY = {vocabulary};

export const MESSAGES = {messages};

export const LANGUAGES = {languages};

export const RTL_LANGUAGES = new Set({rtl});

export function toText(label, language) {{
  const entry = VOCABULARY[label];
  if (!entry) return label;
  return entry[language] || entry.en;
}}

export function message(key, language) {{
  const entry = MESSAGES[key];
  if (!entry) return key;
  return entry[language] || entry.en;
}}

export function isRtl(language) {{
  return RTL_LANGUAGES.has(language);
}}

export function labels() {{
  return Object.keys(VOCABULARY).sort();
}}
"""


def build():
    return HEADER + BODY.format(
        vocabulary=json.dumps(vocabulary.VOCABULARY, ensure_ascii=False, indent=2),
        messages=json.dumps(vocabulary.MESSAGES, ensure_ascii=False, indent=2),
        languages=json.dumps(vocabulary.LANGUAGES, ensure_ascii=False, indent=2),
        rtl=json.dumps(sorted(vocabulary.RTL_LANGUAGES), ensure_ascii=False),
    )


def main():
    enable_unicode()

    generated = build()
    current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else None

    if "--check" in sys.argv:
        if current != generated:
            print(f"{TARGET.relative_to(ROOT)} is out of date.\n"
                  f"Run:  python tools/export_vocabulary.py")
            return 1
        print(f"{TARGET.relative_to(ROOT)} is up to date "
              f"({len(vocabulary.VOCABULARY)} signs, "
              f"{len(vocabulary.MESSAGES)} messages).")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(generated, encoding="utf-8")
    print(f"Wrote {TARGET.relative_to(ROOT)} — {len(vocabulary.VOCABULARY)} signs, "
          f"{len(vocabulary.MESSAGES)} system messages, "
          f"{len(vocabulary.LANGUAGES)} output languages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
