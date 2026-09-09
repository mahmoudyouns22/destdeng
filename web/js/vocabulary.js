/**
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

export const VOCABULARY = {
  "hello": {
    "en": "Hello",
    "ar": "مرحبا",
    "ku": "سڵاو",
    "ku_latin": "Slaw"
  },
  "thank_you": {
    "en": "Thank you",
    "ar": "شكرا",
    "ku": "سوپاس",
    "ku_latin": "Supas"
  },
  "yes": {
    "en": "Yes",
    "ar": "نعم",
    "ku": "بەڵێ",
    "ku_latin": "Bellê"
  },
  "no": {
    "en": "No",
    "ar": "لا",
    "ku": "نەخێر",
    "ku_latin": "Nexêr"
  },
  "please": {
    "en": "Please",
    "ar": "من فضلك",
    "ku": "تکایە",
    "ku_latin": "Tkaye"
  },
  "help": {
    "en": "Help",
    "ar": "مساعدة",
    "ku": "یارمەتی",
    "ku_latin": "Yarmetî"
  },
  "pain": {
    "en": "Pain",
    "ar": "ألم",
    "ku": "ئێش",
    "ku_latin": "Êş"
  },
  "doctor": {
    "en": "Doctor",
    "ar": "طبيب",
    "ku": "پزیشک",
    "ku_latin": "Pizîşk"
  },
  "hospital": {
    "en": "Hospital",
    "ar": "مستشفى",
    "ku": "نەخۆشخانە",
    "ku_latin": "Nexoşxane"
  },
  "medicine": {
    "en": "Medicine",
    "ar": "دواء",
    "ku": "دەرمان",
    "ku_latin": "Derman"
  },
  "water": {
    "en": "Water",
    "ar": "ماء",
    "ku": "ئاو",
    "ku_latin": "Aw"
  },
  "name": {
    "en": "My name is",
    "ar": "اسمي",
    "ku": "ناوم",
    "ku_latin": "Nawim"
  },
  "wait": {
    "en": "Wait",
    "ar": "انتظر",
    "ku": "چاوەڕێ بکە",
    "ku_latin": "Çawerê bike"
  },
  "understand": {
    "en": "I understand",
    "ar": "فهمت",
    "ku": "تێگەیشتم",
    "ku_latin": "Têgeyştim"
  },
  "repeat": {
    "en": "Please repeat",
    "ar": "أعد من فضلك",
    "ku": "تکایە دووبارەی بکەوە",
    "ku_latin": "Tkaye dûbarey bikewe"
  }
};

export const MESSAGES = {
  "not_understood": {
    "en": "Not understood - please sign again",
    "ar": "لم يُفهم - أعد الإشارة من فضلك",
    "ku": "تێنەگەیشتم - تکایە دووبارە ئاماژە بکە",
    "ku_latin": "Têneygeyştim - tkaye dûbare amaje bike"
  },
  "no_hands": {
    "en": "No hands seen - sign inside the frame",
    "ar": "لم تُرَ أي يد - أشِر داخل الإطار",
    "ku": "هیچ دەستێک نەبینرا - لە ناو وێنەکەدا ئاماژە بکە",
    "ku_latin": "Hîç destêk nebînra - le naw wênekeda amaje bike"
  }
};

export const LANGUAGES = {
  "en": "English",
  "ar": "Arabic",
  "ku": "Kurdish"
};

export const RTL_LANGUAGES = new Set(["ar", "ku"]);

export function toText(label, language) {
  const entry = VOCABULARY[label];
  if (!entry) return label;
  return entry[language] || entry.en;
}

export function message(key, language) {
  const entry = MESSAGES[key];
  if (!entry) return key;
  return entry[language] || entry.en;
}

export function isRtl(language) {
  return RTL_LANGUAGES.has(language);
}

export function labels() {
  return Object.keys(VOCABULARY).sort();
}
