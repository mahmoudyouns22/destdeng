# DESTDENG

![Python 3.10 – 3.12](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-3776AB?logo=python&logoColor=white)
![CPU only](https://img.shields.io/badge/runs%20on-CPU%20only-4ECDC4)
![Offline](https://img.shields.io/badge/network-not%20required-4ECDC4)
![Output languages](https://img.shields.io/badge/output-English%20%C2%B7%20Arabic%20%C2%B7%20Kurdish-4ECDC4)
![Self-test](https://img.shields.io/badge/self--test-11%20checks-3DDC84)

**A camera-based sign language recognition system for the Kurdistan Region — turning signs into text, in the language the reader needs.**

*"Destdeng" — from Kurdish **dest** (hand) and **deng** (voice): giving the hand a voice.*

> **Status:** Working end to end on a small vocabulary. Record signs, train, recognise.

---

## The Problem

There are roughly **10,000 deaf people in the Kurdistan Region of Iraq**. As of 2024, there were **five working sign language interpreters** for all of them.

That ratio is the whole problem. It means a deaf person in Erbil who needs to see a doctor, sign a document at a notary office, report something at a police station, or sit an exam usually does it **without an interpreter** — because there is no interpreter to have. Nationally the picture is the same shape: Iraq has over 272,000 deaf citizens and fewer than 100 interpreters, none of them stationed in hospitals or courts. Iraqi Law No. 11 of 2024 guarantees the right to an interpreter. The people to deliver that right do not exist in sufficient numbers.

Three things make this harder than it looks from outside:

1. **Kurdish Sign Language (ZHK) is its own language.** It is *not* mutually intelligible with Iraqi Sign Language, and it is not a signed form of spoken Kurdish. An interpreter trained in Baghdad cannot understand a deaf Kurd from Sulaymaniyah.
2. **It has regional dialects** — Erbil, Sulaymaniyah, and Duhok each developed around their own deaf school, with real lexical differences between them.
3. **Almost no machine learning work exists for it.** The large public sign language datasets are ASL, BSL, and to a lesser extent Arabic Sign Language. For ZHK there is essentially nothing to download. Formal teaching of it only began at university level in 2024.

So the gap is not "another sign language app." Every off-the-shelf sign recognition model in existence is trained on a language the people here do not sign.

## What DESTDENG Does

DESTDENG reads sign gestures from an ordinary webcam or phone camera, recognises them, and renders the meaning as text — with the **output language selected by the reader**: English, Arabic, or Kurdish.

That last part is the design decision that matters. The deaf signer and the person they need to communicate with rarely share a written language either: a deaf Kurdish signer at a hospital may face a doctor who reads Arabic, an NGO worker who reads English, and a clerk who reads Kurdish. One recognition system, three possible outputs, chosen at the moment of use.

**Intended use:** a laptop or phone on a desk between two people — one signing, one reading. No specialised hardware, no gloves, no depth camera, no internet connection required.

## How It Works

```mermaid
flowchart LR
    A[Camera<br/>webcam / phone] --> B[Hand landmark<br/>extraction]
    B --> C[Per-hand normalisation<br/>fixed left/right slots]
    C --> D[Time alignment<br/>trim + resample]
    D --> D2[Gesture classifier]
    D2 --> E2[Novelty + probability gates]
    E2 --> E[Recognised sign<br/>label + confidence]
    E --> F{Output language}
    F --> G[English]
    F --> H[Arabic]
    F --> I[Kurdish]
```

**1. Capture.** Standard camera frames, no special equipment.

**2. Landmark extraction.** Rather than feeding raw pixels to a network, the system extracts 21 hand keypoints per hand, per frame. This is what makes the project feasible on a normal CPU laptop: the model never has to learn "what a hand looks like" from scratch, and it stays robust to skin tone, lighting, clothing, and background — all of which vary enormously in real deployment and none of which carry meaning in sign language.

> **Hands only, for now.** V1 tracks hands and not the upper body. Sign languages place meaning in *where* a sign is made relative to the body as well as in the handshape, so this is a real ceiling on what the current vocabulary can grow into, not a detail. Body pose is on the roadmap below; it is listed as a limitation here rather than glossed over, because the accuracy it costs is easy to mistake for a modelling problem.

**3. Feature construction.** Three things happen here, and each removes a difference that is not the sign ([`src/features.py`](src/features.py)):

- **Per-hand normalisation.** Each hand is translated to put its wrist at the origin and scaled by its own span, so a tall person far from the lens and a child close to it produce comparable features.
- **Fixed hand slots.** MediaPipe returns detected hands in no guaranteed order, so a hand is placed by its *handedness*, never by the order it arrived in. Without this, a two-handed sign looks like two different signs whenever the detector reorders them mid-recording.
- **Time alignment.** Empty frames before and after the sign are trimmed and what remains is resampled to a fixed length. A sign started half a second late fills a different part of the window, and a flattened window is position-sensitive — so without this, *when* you started signing registers as *what* you signed.

**4. Classification.** A Random Forest maps the aligned keypoint sequence, plus per-keypoint movement totals, to a sign label with a probability.

**5. Output.** The recognised label is mapped to its written form in the reader's chosen language. Because the classifier outputs a language-neutral *label*, adding an output language is a data-entry task, not a retraining task.

> **Kurdish means Sorani in Arabic script.** The reader this is built for is in Erbil, Sulaymaniyah or Duhok, and the written Kurdish they read is Sorani — سڵاو, not *Slaw*. Kurmanji in Latin script is the standard in Turkey and Syria; printing it to a Kurdish reader in a clinic in Erbil hands them a transliteration of their own language. The Latin form is kept alongside as `ku_latin`, for anyone who needs it and as a fallback when no Arabic-capable font is installed.

### Design principle: never guess silently

A wrong translation in a hospital is worse than no translation. **The metric that matters is not top-1 accuracy; it is how rarely the system asserts something false with confidence.**

Getting that right took two attempts, and the first one is worth describing because it is a trap.

The obvious approach is to read the classifier's probability and refuse below a threshold. **That does not work.** A Random Forest handed input unlike anything it was trained on still routes it down its trees and can emerge completely certain. Feeding the trained model pure noise produced a **100% confident** answer for a sign nobody made.

So recognition passes through two independent gates (`src/confidence.py`):

| Gate | Question | Catches |
|---|---|---|
| **Novelty** | Is this anything like the data we trained on? Measured as distance to the nearest training sample. | "That wasn't a sign at all" |
| **Probability** | Given it *is* familiar input, is the classifier sure *which* sign? | "That was a sign, but I can't tell which" |

A sample must pass both. They fail in different ways, which is why one alone is not enough. Measured against a trained model: noise, half-scale input and an empty frame are all refused; genuine samples pass at 99%. Those four cases are asserted in the self-test, so the guarantee cannot rot silently.

Two details decide whether the novelty gate is real or decorative:

- **Distance is measured in a reduced space, not on the raw 3,907 features.** In a space that wide every point sits at roughly the same distance from every other, so "nearest neighbour" stops discriminating and the gate quietly waves everything through.
- **The threshold is a quantile, not the maximum.** Keying it to the single most isolated training sample lets one bad recording inflate the limit far enough to admit anything — the gate would still be there, and it would still be useless. The self-test injects exactly that outlier and fails if the limit moves much.

### The refusal must not read as a sign

`Please repeat` is itself a sign in the vocabulary — a deaf person can ask the other party to repeat themselves. So the failure message cannot also be "Please repeat": the reader would have no way to tell *the deaf person asked you to repeat* from *the machine failed*, which is precisely the confident falsehood this project exists to avoid. System messages live in their own table in [`src/vocabulary.py`](src/vocabulary.py), can never be predicted as a label, and say who did not understand.

## Running it

```
pip install -r requirements.txt
python main.py
```

`main.py` is a menu over the whole pipeline, so nothing has to be memorised:

| | |
|---|---|
| **1 Camera test** | Webcam opens, hand skeletons drawn live, FPS counter. Proves the vision layer works. |
| **2 Record samples** | Press SPACE, countdown, make the sign. ~30 samples per sign. |
| **3 Train** | Seconds on a CPU. Prints the score, then explains why that score is optimistic. |
| **4 Recognise** | The system itself. SPACE reads a sign; 1/2/3 switch the output language. |
| **5 Vocabulary** | The signs it knows, in all three languages. |
| **6 Self-test** | The checks below. No camera, no MediaPipe, no OpenCV needed. |

```
python tests/test_pipeline.py
```

Eleven checks over normalisation, hand-slot stability, time alignment, both refusal gates, the vocabulary and the text rendering — almost all on synthetic data, so the pipeline stays verifiable on a machine that cannot install the vision stack at all. Ten of them need only numpy and scikit-learn; the eleventh needs a font to inspect and reports itself as `skip` rather than passing quietly when there is none.

Full instructions and troubleshooting: [`SETUP.md`](SETUP.md).

### What is in this repository

| File | What it does |
|---|---|
| `src/features.py` | Normalisation, handedness slots, time alignment — the feature pipeline. Pure NumPy, so it stays testable without a camera |
| `src/landmarks.py` | The camera test: live keypoint extraction with frame rate and hand-slot readout |
| `src/record_dataset.py` | Records labelled samples into `data/<label>/` |
| `src/train_model.py` | Trains the classifier and fits the novelty check |
| `src/confidence.py` | Decides when **not** to answer — the two gates described above |
| `src/recognise.py` | The live system: camera in, text out, language selectable |
| `src/vocabulary.py` | Label → English / Arabic / Kurdish. The only file that knows about words |
| `src/text_render.py` | Draws Arabic and Kurdish on the video frame (OpenCV cannot), joined and right-aligned. Picks the font and the shaping mode by testing what this machine can actually draw |
| `src/theme.py` | The palette, panels, pills and meters every camera screen is built from — one visual language instead of three |
| `src/console.py` | Lets the terminal print Arabic and Kurdish instead of crashing on them |
| `tests/test_pipeline.py` | The self-test — everything that does not need a camera |
| `src/mp_compat.py` | Fails with a useful message if MediaPipe is the wrong version |

`data/` and `models/` are both gitignored, and the second matters more than it looks: the novelty check carries a compressed projection of the training set inside the model file, so excluding `data/` while publishing `models/` would hand over the recordings anyway.

### The Kurdish that renders is not the Kurdish you typed

Pillow cannot shape Arabic script by itself, so the text is reshaped first and the
letters' joined forms are what reach the font. That indirection hides a trap worth
naming, because it produced broken Kurdish while every check in the project passed.

`arabic-reshaper` has a Kurdish mode, and switching it on looks obviously correct.
It is not: **ە ڵ ێ have no Unicode presentation forms at all**, so the library emits
*private-use* codepoints for them — slots that are empty in every ordinary font.
Tahoma, Segoe UI and Arial all draw them as hollow boxes. سڵاو came out perfectly and
تێنەگەیشتم came out as `ت□نە□گە□یشتم`: the message shown when the system has **not**
understood, rendered unreadably, for the reader least able to check it.

A font-coverage check was already in place and did not catch it, for a reason that
generalises past this bug: **it tested the raw letters.** Every candidate font
contains all 42 characters the vocabulary uses. What the font is asked for is the
reshaper's *output*, and nothing was testing that.

So the font and the shaping mode are now chosen together, by rendering a probe built
from the real vocabulary and comparing pixels — Kurdish mode if some font can draw its
private-use slots, the standard mode otherwise, unshaped before boxes, and the choice
is named in the startup line rather than assumed. Install a Kurdish font that fills
those slots, add it to `FONT_CANDIDATES`, and the better typography is picked up with
no other change. The self-test asserts that nothing on screen is undrawable, so this
cannot come back quietly.

## Scope — Version 1

Ambition without scope is how student projects die. V1 is deliberately narrow:

- A small, fixed vocabulary of **high-frequency practical signs** — the words that come up in a clinic or a government office, not a general-purpose translator
- **Erbil dialect** as the starting dialect, with the architecture kept dialect-agnostic so others can be added
- **Isolated signs**, not continuous signing — recognise one sign at a time before attempting sentences
- **Real-time on CPU** — if it does not run on an ordinary laptop, it does not reach the people who need it

Continuous signing, full grammar, and cross-dialect coverage are explicitly out of scope for V1 and belong on the roadmap.

## Building the Dataset

Since no usable ZHK dataset exists publicly, one has to be recorded. This is treated as part of the engineering work, not a prerequisite obstacle:

- Multiple signers per sign, varied lighting, varied backgrounds, varied distances
- Recorded landmark sequences stored alongside video, so the dataset stays useful if the model architecture changes
- Consent and privacy handled explicitly — contributors know what is recorded and how it is used
- Validation of sign correctness with people who actually use the language, not from a dictionary alone

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python | Ecosystem, and the language the project is developed in |
| Vision / landmarks | OpenCV, MediaPipe | Runs real-time on CPU; no GPU required |
| Modelling | Sequence classifier over keypoint features | Small model, small data requirement, CPU-friendly |
| Interface | Local desktop application | Works offline; nothing leaves the device |

## Roadmap

- [x] Landmark extraction pipeline running live from webcam
- [x] Recording tool for dataset collection
- [x] Baseline classifier + two-gate refusal (novelty and probability)
- [x] Three-language output layer, Kurdish in the script its readers use
- [x] Self-test that runs without a camera
- [ ] Body pose alongside hands — sign location relative to the body carries meaning
- [ ] First dataset pass — core vocabulary, multiple signers
- [ ] Evaluation on unseen signers (not just unseen recordings)
- [ ] Verification of the written vocabulary with Kurdish speakers, and of the signs with deaf signers from the Erbil dialect
- [ ] Field test with deaf users and feedback round
- [ ] Continuous signing (post-V1)
- [ ] Additional dialects: Sulaymaniyah, Duhok (post-V1)

## How This Will Be Evaluated

- **Accuracy on unseen signers** — testing on new recordings of people already in the training set proves nothing about real use
- **False-confident rate** — how often the system displays a wrong sign with high confidence. Target: as close to zero as the vocabulary allows
- **Latency on CPU** — measured on a standard laptop, no GPU
- **Usability with actual deaf users** — whether a real exchange completes, not whether a benchmark number goes up

## Privacy

Video of a person's face and hands is sensitive. DESTDENG processes frames **locally**; recognition does not require sending video anywhere. Dataset recordings are collected with informed consent and are not published without it.

The trained model is **not** a safe stand-in for withholding the data. Its novelty check stores a reduced projection of the training samples, so publishing `models/sign_classifier.pkl` publishes something derived from the people who recorded it. Both directories are gitignored, and the training run says so on its way out.

---

## Author

**Mahmoud Youns**
[github.com/mahmoudyouns22](https://github.com/mahmoudyouns22)

## Sources

- [Kurdish Sign Language — Wikipedia](https://en.wikipedia.org/wiki/Kurdish_Sign_Language) (population, dialects, interpreter count, intelligibility with Iraqi Sign Language)
- [The deaf community in Iraq: Neglect, isolation, and everyday exclusion — Jummar](https://jummar.media/en/2025/03/06/the-deaf-and-mute-in-iraq-neglect-isolation-and-the-barriers-of-language/) (national figures, interpreter shortage, healthcare and legal barriers, Law No. 11 of 2024)
- [Deaf Studies and Kurdish Sign Language Offered for the First Time at AUIS](https://www.auis.edu.krd/News_all/deaf-studies-and-kurdish-sign-language-offered-first-time-auis)
