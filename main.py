"""
DESTDENG — entry point.

A menu over the things this project does, so you can see it work without remembering
any commands.

Run:
    python main.py

THE MENU KNOWS WHAT YOU HAVE NOT DONE YET
    The steps only work in one order — record, then train, then recognise — and
    choosing 4 before 3 produces an error message about a missing model file that
    reads like a broken install rather than a skipped step. So the menu shows what is
    on disk and names the next thing to do, and the entries that cannot work yet are
    dimmed instead of hidden. Dimmed rather than hidden on purpose: a menu whose
    entries appear and disappear gives the reader no idea what the program can do.

    Nothing here is enforced. Every script still runs on its own and still refuses
    with its own explanation; this is a signpost, not a gate.
"""

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "src"
TESTS = BASE / "tests"
PY = sys.executable

sys.path.insert(0, str(SRC))
from console import enable_colour, enable_unicode  # noqa: E402

UNICODE = False
COLOUR = False


def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOUR else text


def dim(text):
    return paint(text, "2;37")


def teal(text):
    return paint(text, "1;36")


def amber(text):
    return paint(text, "33")


def green(text):
    return paint(text, "32")


def rule(width=58):
    return dim(("─" if UNICODE else "-") * width)


def run(script, *args):
    """Run one of the project's scripts.

    cwd is pinned to the project root on purpose: the scripts resolve data/ and
    models/ relative to the project, and leaving cwd wherever the user happened to
    launch from is how a dataset ends up split across two directories.
    """
    subprocess.run([PY, str(script), *args], cwd=str(BASE))


def ask_count(prompt, default=30):
    """Read a positive integer, or fall back to the default. Never raises."""
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"'{raw}' is not a number — using {default}.")
        return default
    if value < 1:
        print(f"Needs to be at least 1 — using {default}.")
        return default
    return value


def survey():
    """What is on disk: how many samples, across how many signs, and a model or not."""
    data = BASE / "data"
    model = BASE / "models" / "sign_classifier.pkl"

    signs = samples = 0
    if data.exists():
        for directory in data.iterdir():
            if directory.is_dir():
                found = len(list(directory.glob("*.npy")))
                if found:
                    signs += 1
                    samples += found

    return {"signs": signs, "samples": samples, "trained": model.exists()}


def next_step(state):
    """The one thing worth doing next, given what is on disk."""
    if state["signs"] < 2:
        return 2, "record a second sign — the classifier needs two to tell apart"
    if not state["trained"]:
        return 3, "train the classifier on what you have recorded"
    return 4, "run recognition"


ENTRIES = [
    ("1", "Camera test", "see hand keypoints live", lambda s: True),
    ("2", "Record samples", "build the dataset", lambda s: True),
    ("3", "Train", "fit the classifier", lambda s: s["signs"] >= 2),
    ("4", "Recognise", "the actual system", lambda s: s["trained"]),
    ("5", "Vocabulary", "the signs it knows, in all three languages", lambda s: True),
    ("6", "Self-test", "no camera needed", lambda s: True),
]


def draw(state):
    step, advice = next_step(state)

    print()
    print("  " + teal("DESTDENG") + dim("   sign language recognition"))
    print("  " + rule())

    if state["samples"]:
        dataset = "{} samples across {} sign{}".format(
            state["samples"], state["signs"], "" if state["signs"] == 1 else "s")
    else:
        dataset = dim("nothing recorded yet")
    model = green("trained") if state["trained"] else dim("not trained yet")

    # Padded before colouring, never after. An escape sequence is characters as far
    # as str.format is concerned, so f"{dim('dataset'):<18}" pads to 18 including the
    # nine invisible ones and the column stops lining up the moment colour is on.
    print(f"  {dim('dataset'.ljust(16))}  {dataset}")
    print(f"  {dim('model'.ljust(16))}  {model}")
    print()

    for key, name, note, available in ENTRIES:
        ready = available(state)
        marker = teal("▸") if UNICODE and key == str(step) else " "
        row = f"  {marker} {key}  {name:<16} {note}"
        print(row if ready else dim(f"    {key}  {name:<16} {note}"))

    print(dim("      q  Quit"))
    print()
    print("  " + amber("next:") + f" {advice}")
    print()


def main():
    global UNICODE, COLOUR
    UNICODE = enable_unicode()
    COLOUR = enable_colour()

    while True:
        state = survey()
        draw(state)

        choice = input("  > ").strip().lower()

        if choice == "1":
            run(SRC / "landmarks.py")

        elif choice == "2":
            label = input("  Sign label (e.g. hello): ").strip()
            if not label:
                print("  Need a label.")
                continue
            count = ask_count("  Target total samples for this label [30]: ")
            run(SRC / "record_dataset.py", "--label", label, "--samples", str(count))

        elif choice == "3":
            run(SRC / "train_model.py")

        elif choice == "4":
            run(SRC / "recognise.py")

        elif choice == "5":
            run(SRC / "vocabulary.py")

        elif choice == "6":
            run(TESTS / "test_pipeline.py")

        elif choice == "q":
            break

        else:
            print("  Pick 1-6 or q.")


if __name__ == "__main__":
    main()
