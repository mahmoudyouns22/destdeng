"""
vocabulary.py — the sign vocabulary and its written form in each output language.

The classifier never outputs text. It outputs a LABEL, which is language-neutral.
This file is the only place that maps a label to words. That is what makes adding a
fourth output language a data-entry task rather than a retraining task.

To add a sign:
  1. Add an entry to VOCABULARY here.
  2. Record samples:  python src/record_dataset.py --label <key> --samples 30
  3. Retrain:         python src/train_model.py

Starter vocabulary: words that come up in a clinic or a government office, which is
where the absence of an interpreter costs the most.

WHY THE KURDISH COLUMN IS SORANI IN ARABIC SCRIPT
    The reader this project is built for is in Erbil, Sulaymaniyah or Duhok, and the
    written Kurdish they read is Sorani in Arabic script. Kurmanji in Latin script is
    the standard in Turkey and Syria, not here — printing "Supas" to a Kurdish reader
    in a clinic in Erbil hands them a transliteration of their own language. The
    Latin form is kept in `ku_latin` for anyone who does need it, and because it
    stays readable when no Arabic-capable font is installed.

WHAT STILL HAS TO BE VERIFIED BY PEOPLE
    - The written words below must be checked with a Kurdish speaker before the
      system is used with anyone.
    - The Kurdish Sign Language signs themselves must be validated with deaf signers
      from the Erbil dialect — not taken from a dictionary.
    Neither of those is something code can settle.

SYSTEM MESSAGES ARE NOT VOCABULARY
    MESSAGES is deliberately separate from VOCABULARY. A message the system shows
    about itself ("I did not understand") must never be confusable with a sign the
    signer actually made. See the note above MESSAGES.
"""

VOCABULARY = {
    "hello":       {"en": "Hello",           "ar": "مرحبا",        "ku": "سڵاو",                  "ku_latin": "Slaw"},
    "thank_you":   {"en": "Thank you",       "ar": "شكرا",         "ku": "سوپاس",                 "ku_latin": "Supas"},
    "yes":         {"en": "Yes",             "ar": "نعم",          "ku": "بەڵێ",                  "ku_latin": "Bellê"},
    "no":          {"en": "No",              "ar": "لا",           "ku": "نەخێر",                 "ku_latin": "Nexêr"},
    "please":      {"en": "Please",          "ar": "من فضلك",      "ku": "تکایە",                 "ku_latin": "Tkaye"},
    "help":        {"en": "Help",            "ar": "مساعدة",       "ku": "یارمەتی",               "ku_latin": "Yarmetî"},
    "pain":        {"en": "Pain",            "ar": "ألم",          "ku": "ئێش",                   "ku_latin": "Êş"},
    "doctor":      {"en": "Doctor",          "ar": "طبيب",         "ku": "پزیشک",                 "ku_latin": "Pizîşk"},
    "hospital":    {"en": "Hospital",        "ar": "مستشفى",       "ku": "نەخۆشخانە",             "ku_latin": "Nexoşxane"},
    "medicine":    {"en": "Medicine",        "ar": "دواء",         "ku": "دەرمان",                "ku_latin": "Derman"},
    "water":       {"en": "Water",           "ar": "ماء",          "ku": "ئاو",                   "ku_latin": "Aw"},
    "name":        {"en": "My name is",      "ar": "اسمي",         "ku": "ناوم",                  "ku_latin": "Nawim"},
    "wait":        {"en": "Wait",            "ar": "انتظر",        "ku": "چاوەڕێ بکە",            "ku_latin": "Çawerê bike"},
    "understand":  {"en": "I understand",    "ar": "فهمت",         "ku": "تێگەیشتم",              "ku_latin": "Têgeyştim"},
    "repeat":      {"en": "Please repeat",   "ar": "أعد من فضلك",  "ku": "تکایە دووبارەی بکەوە",  "ku_latin": "Tkaye dûbarey bikewe"},
}

# Things the SYSTEM says about itself. Kept out of VOCABULARY on purpose.
#
# "repeat" is a sign in the vocabulary: a signer can genuinely ask the other person to
# repeat themselves. If a failed recognition also printed "Please repeat", the reader
# could not tell "the deaf person asked you to repeat" from "the machine failed" —
# which is exactly the kind of confident falsehood this project exists to avoid. So
# the refusal wording is distinct, and says who did not understand.
MESSAGES = {
    "not_understood": {
        "en": "Not understood - please sign again",
        "ar": "لم يُفهم - أعد الإشارة من فضلك",
        "ku": "تێنەگەیشتم - تکایە دووبارە ئاماژە بکە",
        "ku_latin": "Têneygeyştim - tkaye dûbare amaje bike",
    },
    "no_hands": {
        "en": "No hands seen - sign inside the frame",
        "ar": "لم تُرَ أي يد - أشِر داخل الإطار",
        "ku": "هیچ دەستێک نەبینرا - لە ناو وێنەکەدا ئاماژە بکە",
        "ku_latin": "Hîç destêk nebînra - le naw wênekeda amaje bike",
    },
}

LANGUAGES = {
    "en": "English",
    "ar": "Arabic",
    "ku": "Kurdish",
}

# Languages written right to left. text_render.py needs this to align them correctly.
RTL_LANGUAGES = {"ar", "ku"}


def _pick(entry, language):
    if entry is None:
        return None
    return entry.get(language) or entry["en"]


def to_text(label, language="en"):
    """Return the written form of a recognised label in the chosen language."""
    text = _pick(VOCABULARY.get(label), language)
    return label if text is None else text


def message(key, language="en"):
    """Return a system message — never a sign. See the note above MESSAGES."""
    text = _pick(MESSAGES.get(key), language)
    return key if text is None else text


def is_rtl(language):
    return language in RTL_LANGUAGES


def labels():
    """Every label the system knows about."""
    return sorted(VOCABULARY.keys())


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from console import enable_unicode

    enable_unicode()

    print(f"{len(VOCABULARY)} signs in the vocabulary:\n")
    for key in labels():
        row = VOCABULARY[key]
        print(f"  {key:<12} EN: {row['en']:<16} AR: {row['ar']:<14} KU: {row['ku']}")

    print("\nSystem messages (not signs, never predicted):\n")
    for key in sorted(MESSAGES):
        print(f"  {key:<16} {MESSAGES[key]['en']}")
