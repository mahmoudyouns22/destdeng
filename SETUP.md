# Setup

Tested on Windows with an ordinary laptop webcam. No GPU required.

## 1. Python — this is the step that goes wrong

You need **Python 3.10 – 3.12**. MediaPipe 0.10.14 publishes wheels for 3.8 – 3.12
only, so on 3.13 or newer `pip install` fails with *"no matching distribution"* — which
reads like a broken requirements file rather than a Python that is too new.

```
python --version
```

Having a newer Python installed is fine; you just have to build the virtual environment
with a supported one. On Windows, check what you have and pick:

```
py -0p
```

If 3.12 is not in the list, install it from [python.org](https://www.python.org/downloads/)
and carry on — it will sit alongside whatever you already have without disturbing it.

## 2. Virtual environment

```
py -3.12 -m venv .venv
.venv\Scripts\activate
```

(macOS / Linux: `python3.12 -m venv .venv` then `source .venv/bin/activate`)

## 3. Install

```
pip install -r requirements.txt
```

## 4. Check it before touching a camera

```
python tests/test_pipeline.py
```

Eleven checks over the feature pipeline, both refusal gates, the vocabulary and the
text rendering. Ten of them need only numpy and scikit-learn — no OpenCV, no
MediaPipe, no webcam — so this also works as a way to verify the logic on a machine
where the vision stack will not install. The eleventh needs a font to inspect and
prints `skip` there rather than passing quietly. Nothing should print FAIL.

## 5. Run

```
python main.py
```

A menu appears. Work through it in order:

**1 — Camera test.** Your webcam opens and hand skeletons are drawn over your hands,
with a live FPS counter and a `slots` readout. `slots ##` means both hands were
assigned to their feature slots, `#-` means one hand. If you see this, the whole vision
layer works.

**2 — Record samples.** Pick a label from the vocabulary (`hello`, `thank_you`, `yes`,
`no`, `help`, `pain`, `doctor`, …). Press SPACE, wait for the countdown, make the sign.
Do about 30 samples. Then record a second sign — the classifier needs at least two to
have anything to tell apart.

The sample count is a **target total**, not a number to add: recording is resumable, so
running it again with the same label tops the folder up rather than starting over.

**3 — Train.** Takes seconds on a CPU. It prints how it scored on held-out samples, and
then tells you plainly why that score is optimistic.

**4 — Run recognition.** The real system. Press SPACE to read a sign; press 1, 2 or 3
to switch the output language between English, Arabic and Kurdish.

**6 — Self-test.** The same checks as step 4 above, from the menu.

The menu also reads `data/` and `models/` each time it is drawn, shows what is on
disk, dims the steps that cannot work yet, and names the next one worth doing — so
the order above does not have to be remembered either.

## Troubleshooting

**"no matching distribution found for mediapipe"** — your Python is 3.13 or newer. See
step 1; you need a 3.12 virtual environment.

**"Could not open the camera"** — another app is holding it. Close Zoom, Teams, the
Camera app, and any browser tab with camera permission.

**FPS below 15** — close other apps. The model is already set to the lightest setting
(`model_complexity=0`).

**Arabic or Kurdish shows as boxes, backwards, or with the letters unjoined** — the
shaping libraries or a suitable font are missing. Recognition still works. The startup
line in step 5 tells you exactly what is missing, and any font or shaping problem is
reported there as a `warning:` line rather than silently producing broken text.

**Kurdish shows boxes but Arabic is fine** — your font has Arabic but not the Sorani
letters (ڕ ڵ ۆ ێ). Tahoma or Segoe UI on Windows cover them; the program picks the
first installed font that does and says which one it chose.

**"This model was trained on feature layout v1"** — the feature pipeline changed since
you trained. Retrain: `python src/train_model.py`. This is deliberate; scoring new
features against an old model would produce confident nonsense.

**"No trained model"** — you skipped step 3, or you have fewer than 2 signs recorded.

**Arabic crashes the terminal with `UnicodeEncodeError`** — should not happen any more;
the entry points switch stdout to UTF-8 on startup. If it does, you are running a
script directly in a way that bypasses that, so report it.

## What is not in this repository

`data/` (your recordings) and `models/` (the trained file) are both gitignored.

Recordings are video-derived data of real people and are not published. The **model is
not a safe substitute** for withholding them: its novelty check stores a reduced
projection of the training set, so sharing the `.pkl` shares data derived from the
people who recorded it. Train from the data instead of passing the model around.
