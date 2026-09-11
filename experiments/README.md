# experiments

Not part of the engine. Kept because the work is real and the findings are
worth not repeating, but neither of these is wired into `physicaldrum/`.

### `snare_wires.py` — a one-sided contact model for snare wires

Works, in the sense that the contact solves stably and produces physically sized
forces (0.5–1.8 N per strand, right for a snare). Two things had to be correct
for that: the contact must be solved **implicitly** (head pushes wire, wire
pushes head, both from last sample's values, against a stiff spring = immediate
blow-up), and the injection needs a `dt`, because the impulse-invariant input
takes an impulse and a contact force is newtons. Without that `dt` every
collision was crushed by a factor of 44100 and the peak contact force read 0.00 N.

**But it does not make a snare sound**, and the reason is structural rather than
a tuning problem. It rattles against a single slow breathing mode, because the
cavity in the current engine is one lumped compliance — so `INT INT Phi dA = 0`
makes batter→reso coupling identically zero at every angular order above 0. The
wires have nothing textured to bounce off. Spectral peak-to-mean stays at 35–48
dB where noise is 8–14, and damping the contact removes the ring and all the
high-frequency energy together, which is the proof that the energy was the
lumped contact spring resonating rather than real collisions.

Fixing it means `cavity_modes.py` below.

### `cavity_modes.py` — the cavity's real mode set

The fix for the above. Above `1.8412 c/(2 pi a)` — 565 Hz for a 14" drum — the
pressure inside is not uniform and the two sheets couple at every angular order.
The overlap integral is Lommel again, the same closed form as the far field and
the added mass, and the current rank-one model is its `q = s = 0` term.

**Half built, and its eigenvectors are not trustworthy** — see the module
docstring. The eigenvalues check out exactly against rank-one; the vectors do
not, because it is a non-symmetric generalized problem.

### `_legacy_engine.py`

The pre-stiffness Python engine, kept only because `snare_wires.py` was written
against it. Do not build on it.
