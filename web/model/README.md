# `web/model/`

Drop a signs file here as **`signs.json`** and the site loads it at startup: every
visitor gets a system that already recognises those signs, instead of one that knows
nothing until they spend a quarter of an hour teaching it.

Produce one from the site itself — **Teach signs → Teach every sign**, then **Save
signs to a file** — and commit the result as `web/model/signs.json`.

Nothing is here yet, and that is not a bug. A deployment without this file works: the
page says it has not been taught anything and offers to learn.

---

### Why there is no file in the repository

There is no public dataset for Kurdish Sign Language. That absence is the reason this
project exists and is documented in the main README — the large sign language corpora
are ASL, BSL and Arabic Sign Language, and a model trained on any of them is trained
on a language the people here do not sign.

A file could be generated without a camera. It must not be. The samples would be
invented numbers, and the result would be a system answering with confident words for
gestures that mean nothing — the single failure this project is engineered to prevent,
in the place it is least likely to be caught. Synthetic data is used in the test
suites, where it is labelled as such and never leaves them.

So the file comes from someone signing in front of a camera, or it does not exist.

### Before committing one

The samples are hand landmark coordinates, not video — there is no image of anyone in
the file. It is still derived from a person's hands, and publishing it publishes
something of theirs. Commit one recorded by you, or one whose signers agreed to it.

And the signs themselves need validating with deaf signers from the dialect they are
meant for. A file recorded by a hearing person copying a description teaches the
system that description, confidently, forever.
