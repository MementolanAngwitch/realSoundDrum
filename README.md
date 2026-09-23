# physicalDrumBeta

A struck sheet, solved from its physical properties. No samples, no fitted envelopes,
no oscillator banks tuned by ear — a clamped circular sheet is given a material, a size,
a tension and a thickness, air is put behind it, and something with its own physical
properties hits it. What comes out is whatever that is.

The primitive is deliberately **unnamed**. Some settings are a tom; some are a sheet of
paper; some are an aluminium plate. Fitting the model to a particular instrument is a
separate job, done afterwards, against a model that does not already assume the answer.

---

## Quick start

```bash
pip install -r requirements.txt     # numpy, scipy -- nothing else
python build_tables.py              # generates the two tables (~1 min)
python render.py                    # renders every preset to .wav
python tests/test_parity.py         # Python vs the browser engine
python tests/test_physics.py        # against exact results and measurements
```

Python 3.7 or newer, on any OS. Only numpy and scipy, no compiler, no system
libraries, no network. `build_tables.py` must run once before anything else —
the two tables it writes are generated, not source, and every other entry point
fails with an explicit message telling you to run it if they are absent.
Everything after that is offline and self-contained.

For the browser engine, open `web/physicalDrumBeta.html` — one file, no server,
no build step, no dependencies. Double-click it. The engine, the mode table and
the air table are all inlined; the only network request the page makes at all is
a Google Fonts stylesheet, and it falls back to system fonts if that is blocked,
so it works fully offline. It can also be served straight from the repo with
GitHub Pages — see [Putting it online](#putting-it-online).

```python
from physicaldrum import Params, build, render, PRESETS

d = build(Params(mat='Steel', a=0.30, T=9000, h=6.0, d=3000, Tr=0))
y, r = render(d, impulse=0.028)
print(d.f1, d.info['stiffness'], r['bend'])
```

### Two implementations, deliberately

The Python package and the browser page are independent implementations of the
same physics. That is worth keeping: a disagreement is a bug in one of them, and
it is how several real ones were found. They drifted apart once — the browser
gained bending stiffness, materials, the in-plane tension lag and the (q,v)
integrator while the Python kept none of them, and nothing anywhere failed —
so `tests/test_parity.py` now fails if they diverge.

Current agreement, across six cases spanning Mylar to steel to latex, a kettle, and
a snare with and without its wires:

| | |
|---|---|
| Mode frequencies | within 2.4e-5 relative |
| f1, tau, stiffness share, air fraction, beta, tau_L | within 5e-3 |
| gamma, pitch bend, peak dent, limiter drive | within 5e-3 |
| Waveform, first 50 ms (linear cases) | −66 dB or better |
| Band energies, three windows, full rate | within 0.12 dB (linear), 0.21 dB (wires) |
| When the strands settle | same sample |

The waveform is compared over the onset rather than the whole render because the
browser's mode table is rounded to 4 dp to keep the page small. That leaves ~1e-5
in every frequency, which over 1.45 s is up to a quarter cycle of phase at 16 kHz
— a large sample-by-sample residual that says nothing about the physics. The band
comparison is the statement that survives it.

The wired snare gets no waveform comparison at all. A rattle is a nonlinear, partly
chaotic process, and two engines that start 1e-5 apart part company within
milliseconds, exactly as two real snares would. What must agree is where the energy
goes and when the strands settle, and both do.

---

## The model

The chain, in the order the code applies it.

### 1. Modes

A clamped circular sheet of radius `a` has mode shapes

```
Φ_mn(r,θ) = J_m(j_mn · r/a) · cos(mθ)          j_mn = the n-th zero of J_m
```

`m ≥ 1` modes are doubly degenerate (a `cos mθ` and a `sin mθ` partner). Gauge-fixing them
to the strike axis is valid for **one** strike point only.

Normalisation matters and is easy to get wrong:

```
∬ Φ² dA = ε_m · (πa²/2) · J_m′(j_mn)²          ε_m = 2 if m = 0 else 1
```

Store `J_m′(j_mn)` **signed**. `sqrt` of the squared value loses the sign, and the sign is
physical — it sets the relative phase of the partials.

### 2. Frequency: tension *and* bending stiffness

```
ω² = (T k² + D k⁴) / σ_eff        k = j_mn/a,   D = E h³ / 12(1−ν²)
```

The `k⁴` term is not a correction. On a 10 mil Mylar head it is already half the restoring
force by `j = 120`, and 98% on a steel foil:

| j | T k² | D k⁴ | stiffness share |
|---|---|---|---|
| 10 | 1.29e+07 | 8.99e+04 | 0.7% |
| 80 | 8.26e+08 | 3.68e+08 | 30.8% |
| 120 | 1.86e+09 | 1.86e+09 | 50.1% |
| 170 | 3.73e+09 | 7.51e+09 | 66.8% |

A model with only `T k²` can only ever be a drum. **Keep the two terms in separate columns**
all the way through — tension modulation scales only the first, and the interface contract
depends on it.

Caveat: the mode *shapes* used are the membrane ones. A stiffness-dominated sheet is
therefore approximated, not solved. The **Stiffness at top** readout says how far into that
approximation you are.

### 3. Air added mass

A moving sheet drags a layer of air with it. In the Hankel domain the mode's transform is
the same closed form as its far field, and the reactive part is the tail beyond `u = ωa/c`:

```
added σ (one open face) = ρ_air · a · A
A = 2j² · ∫₀^∞ [ J_m(u)/(u²−j²) ]² ds ,   u = √(s² + (rj)²) ,   r = c_phase/c_air
```

- The `s` substitution is not cosmetic. A naïve grid in `u` across the singularity gives the
  (1,1) mode a 1071-cent drop instead of 507.
- **Which faces.** A sheet over a sealed cavity has one open face; its cavity face is the
  cavity's job (§4). A sheet with nothing behind it has room air on both faces for `m ≥ 1`,
  but only one face's worth for `m = 0`: air flows round the rim of an unbaffled sheet, and
  for breathing motion that relief is total — Lamb's rigid disc carries `8ρa³/3` moving
  broadside in free air, exactly one baffled face.
- **Two calibrations, and you need both.** The rigid baffled piston, `M = 8ρa³/3`, fixes the
  constant for `m = 0` to six decimal places. The infinite-plane limit fixes it for every
  order: far above the lowest modes a pattern of wavenumber `k` carries one air layer `1/k`
  thick, so `A·j → 1` for every `m`.

That second calibration found a factor of two. The formula once read `4j²/ε_m`, dividing by
the angular factor a second time — right for `m = 0`, exactly twice too heavy for every
`m ≥ 1` (`A·j` came out 2.03). The piston could not see it, because the piston is `m = 0`.
It had been hiding in plain sight: two faces' worth of air through one face is roughly what a
sealed kettle does, so the old timpani check still passed.

This is the term that makes a drum a *pitched* instrument: it loads the low modes hardest,
which drags the Bessel ratios toward harmonic.

### 4. The cavity

The air between the sheets has modes of its own. In a rigid cylinder of radius `a` and depth `L`:

```
Ψ_mqs = J_m(χ_mq r/a) · cos(mθ) · cos(sπx/L)        χ_mq = zeros of J_m′  (rigid wall)
ω_mqs = c · √((χ/a)² + (sπ/L)²)
```

A sheet mode couples only to cavity modes of the **same** `m`, so everything is block
diagonal in angular order. The overlap is Lommel again — the fourth use of one integral,
after the norm, the far field and the added mass:

```
G = ∬ Φ Ψ dA = ang · a² · (−j J_m′(j) J_m(χ)) / (j² − χ²)
```

`χ = 0, m = 0` gives `∬ Φ dA`: the old one-compliance model is the `q = s = 0` term of this.
It coupled only the breathing modes, so the struck sheet could not drive the far one at any
angular order above zero. That is why snare wires had nothing to rattle against.

**The symmetric form.** Projected directly, the coupling sits in the stiffness one way and
the mass the other, and an eigensolver returns vectors that are not mass-orthogonal. In the
variable `η_n` — the volume of air the cavity mode has actually moved —

```
M q″ + [K + Σ κ g gᵀ] q − Σ κ g η = 0          κ_n = ρc²/Λ_n
μ_n η″ + κ_n η_n − κ_n gᵀ q      = 0          μ_n = κ_n/ω_n²
```

the mass is diagonal, the stiffness symmetric, and a plain symmetric eigensolver does it
(Cholesky + Jacobi in the browser, which has no LAPACK). For the uniform mode `ω = 0`,
`μ → ∞`, `η` stays at zero, and what is left is `κ₀ R Rᵀ` — the rank-one model exactly.
`tests/test_physics.py` checks that limit against the secular equation to 1e-6.

**Two kinds of cavity mode.** Below `F_DYN` — 5 kHz, or 3× the fastest coupled sheet mode,
whichever is lower — a cavity mode is a real degree of freedom. Above it the air cannot keep
up and the mode leaves only an added mass `μ g gᵀ`, which summed over every axial order is
closed form:

```
Σ_s μ_s g gᵀ = ρ G Gᵀ / (ang · rad · α) × { coth(αL)  same sheet
                                          { csch(αL)  across the cavity        α = χ/a
```

The `csch` is the evanescent field crossing the drum: a pattern of radial wavenumber `α` is
felt at the far sheet with weight `e^(−αL)`, which is why only broad patterns get through.
The `coth → 1` limit is a half-space, and there it agrees with the open-air calculation of §3
to 2–7.5% — two independent routes to one quantity, the cylinder always slightly heavier
because its wall stops air escaping over the rim.

Sheet patterns finer than `J_CAV` (at least `4a/L`) cannot reach the far sheet at all; they
see the cavity as a half-space and get the open-air layer on that face too.

The vent is a slug of air in the shell wall, mass `ρ_air A_p L_eff`, seeing only the
uniform pressure; the Helmholtz resonance falls out.

**Validation.** Through a full kettle — one sheet, open air on top, the cavity's modes
beneath — the timpano's partial ratios against Rossing's measurements, nothing fitted:

```
ideal membrane   1  1.34   1.665  1.98   2.289     21.0% max error
this model       1  1.523  2.028  2.522  3.010      4.1% max error
measured         1  1.5    1.99   2.44   2.89
```

### 5. Snare wires

Strands lie across the second sheet, perpendicular to the strike axis so the whole
configuration stays symmetric about it (the mode set is `cos mθ` only, which is then exact).
Each strand is a string of tension `T` and density `μ`, anchored beyond the rim. What holds it
on is the **snare bed**: the anchors sit a depth `bed` past the sheet's plane, so the whole
preload is carried where the strand bends over the beds, and over the flat middle a straight
strand bears on the sheet with nothing but its neighbours' tension.

**The contact is a constraint, not a spring.** Every sample, each point either touches with a
pushing force and no overlap, or is clear with no force:

```
F ≥ 0 ,   η − A F ≤ 0 ,   F · (η − A F) = 0
```

`A` is the one-step compliance between every pair of points — the sheet's, from its modes,
plus each strand point's `Δt²/m`. It is symmetric positive definite, so the solution is
unique: Python finds it by active set, the browser by the same pivoting with Schur
complements on `A⁻¹`, and both land on the same forces. There is no contact stiffness to
choose — a steel coil is ~9e8 N/m per metre radially, a 4 µs contact, far below one sample,
so at audio rate the strand is rigid and the bounce is the sheet's own elasticity.

Where the strands start matters: pressed on, at the integrator's exact fixed point under
their own preload. Starting from zero fires the preload as a step — a thump that is not in
the drum.

Once the sheet under the strands has fallen 60 dB below its peak (in acceleration), nothing
the wires do is audible, and they are left resting with their last forces on. The residual
against never doing so is −58.6 dB; it saves most of the cost.

**What it gives.** At the snare preset's strike the strands leave the sheet within the first
milliseconds, rattle for ~40 ms, and hold 2–16 kHz ~4.5 dB above the unsnared drum 60–150 ms
after the hit, when the sheet's own highs have died; after that the resting strands damp the
far sheet, as snares do. That is converged — the same to within a fraction of a dB from 5 to 8
points per strand and a 10 to 14 kHz mode ceiling. The rattle lengthens with a harder hit
(60 ms at 3×, 160 ms at 10×) and **that is the open question**: a hit hard enough to buzz like
a real snare drives the tension model to its clamp. See *Not modelled*.

### 6. Radiation

The Rayleigh far field of a baffled disc, which for a *clamped* mode has a closed form
because `J_m(j_mn) = 0` at the rim:

```
D_mn = j · J_m′(j) · J_m(β) / (β² − j²) ,     β = (ωa/c_air) · sin θ
```

- `∬ Φ dA` is **not** "radiation" — it is the `ka → 0` monopole limit of this, and it is
  exactly zero for every `m ≥ 1`. Using it silences most of the mode set.
- At `θ = 0` the expression does vanish for `m ≥ 1`: an on-axis mic really does hear only
  the breathing modes. Off axis it does not.
- `β → j` is **coincidence**, where the sheet's phase speed reaches the speed of sound. The
  expression is 0/0 there and the limit is `πa² J_m′(j)²` — finite and large. That is a real
  radiation peak, and it is why a cymbal is loud and a drumhead is not. A steel foil reaches
  it at `j = 144`, inside the audio band.

### 7. Damping

```
σ_i = σ₀ + η · ω_i / 2
```

A constant loss factor — constant Q. A polymer loses a fixed *fraction* of the strain energy
per cycle, not a fixed amount per second. `η ≈ 0.015` for Mylar lands inside the measured
t60 range for a real tom in every octave from 200 Hz to 3 kHz.

`σ = σ_a + σ_b f²` is a fitted curve with no mechanism, and measured against a render it is
~3.5× too weak at 200 Hz and ~2× too strong at 3 kHz — low partials sustaining while the
highs vanish, which is what "metallic" means.

### 8. The strike

An impulse `P` delivered through a Gaussian contact patch of radius `w`, over a contact time

```
τ = 2w / c              floored by the striker's own softness (felt, rubber)
```

Contact time is **not a free knob** — the dent spreads at the sheet's wave speed, so the tip
stays down about as long as the sheet needs to get out from under it. Tip radius is the
single biggest control on brightness, because it band-limits the strike twice over: through
the spatial roll-off `exp(−(kw)²/2)` and through `τ`.

### 9. Tension modulation

Denting a sheet stretches it, which raises its tension, which raises every tension-borne
partial and then relaxes:

```
S = ∬ |∇u|² dA = Σ k_i² q_i² norm_i
γ = √(1 + β S)              β = E h / (2π a² T)
ω² → γ² · ω_T² + ω_D²       γ multiplies ONLY the tension term
```

`S` sums every mode of the struck sheet, the breathing (`m = 0`) ones included. An earlier
version left those out — and on a centre hit they are most of the stretch. The peak-dent
readout had the same gap, which is why the bass drum preset had been calibrated to a
"3.25 mm" dent that was really 14.45 mm; it is re-anchored to 4 mm.

**The tension rise is not instantaneous.** It is carried across the sheet by the in-plane
wave at `c_L = √(E/ρ)` — 1700 m/s in Mylar, 5100 in aluminium — so global tension lags local
stretch by about one crossing time, `τ_L = 2a/c_L`. Treating it as instantaneous lets the
first few samples, when every mode is still in phase and the modal sum for `S` is at its most
coherent and least converged, set γ for the whole note. On a stiff sheet that spike alone
pinned γ at its clamp: 74 cents at `P = 5 mN·s`, 814 at 7.18, nothing in between. One pole
with `τ_L` removes it.

### 10. Integration

Per mode, state is **(q, v)** — position and velocity — advanced by the exact damped-oscillator
rotation:

```
q′ = e^(−σΔt) [ q·cos(ω_dΔt) + (v + σq)/ω_d · sin(ω_dΔt) ]
v′ = e^(−σΔt) [ v·cos(ω_dΔt) − (ω²q + σv)/ω_d · sin(ω_dΔt) ]
```

The two-pole form `q[n+1] = a1·q[n] − a2·q[n−1]` is exact at a *fixed* frequency and is what
to use when there is no modulation. But swapping `a1` while keeping both past samples is not
energy-consistent: the pair encodes an amplitude that means something different under the new
frequency. Position and velocity are physical, so the same rotation with a new ω is simply
the same state seen by a different oscillator. The strike is then just a velocity kick.

---

## Things that will bite

Every one of these was found the hard way.

**`∬ Φ dA` is a limit, not a definition.** It is the `ka → 0` monopole term. Using it as the
radiation weight silences every `m ≥ 1` mode — 187 of 200 on a typical head.

**Damping laws need a mechanism.** If you cannot say what physical process a damping term
represents, it is a curve fit, and it will be wrong somewhere in the band.

**Band ceilings are invisible.** A mode table that stops at `j < 82` caps a 16" tom at
7.9 kHz. A real close-miked drum has usable energy to 12–16 kHz. Nothing errors; the sound
is just dull.

**Prune by radiation, not by frequency.** Reaching 16 kHz naively needs ~2300 modes, but
`J_m(β)` is exponentially small once `m ≫ β`, so high angular orders are acoustically
invisible — only ~290 of those 2300 clear −50 dB. Keep those plus everything below a low `j`
cut (quiet, but they carry `S`), and the band triples for ~300 extra modes.

**Order modes by lifetime.** With constant Q a 10 kHz mode is inaudible after ~25 ms while
the fundamental runs for a second. Sorting by death sample and shrinking an active count is
~15× and it is exact, not an approximation.

**`S` only converges once the band resolves the striker tip.** A point force on a membrane
has a 1/r slope, so `∬|∇u|² dA` diverges logarithmically; the tip radius regularises it, but
only if modes out to `j ~ 3a/w` are present. Measured on a 16" tom:

| tip | 200 modes | 500 | 1000 | 1500 | spread |
|---|---|---|---|---|---|
| 10.0 mm | 1.0220 | 1.0220 | 1.0220 | 1.0220 | 0% |
| 3.5 mm | 1.1230 | 1.1484 | 1.1588 | 1.1588 | 3% |
| 1.5 mm | 1.1660 | 1.2311 | 1.3973 | 2.0000 | **72%** |

Below that, the pitch bend is a truncation artifact. Floor the tip at what the retained band
can see, and say so.

**γ can push modes past Nyquist.** Once the band reaches 16 kHz, a γ of only 1.38 folds the
top modes to arbitrary low frequencies — an audible screech. Clamping them *to* Nyquist is no
better: hundreds then pile onto one frequency and ring in unison. Simply do not modulate
modes that could ever exceed it; they are above 13.8 kHz, where a few hundred cents of glide
is inaudible.

**Impulse is the wrong thing to hold fixed.** Anchor calibration on something physical and
measurable — a 4 mm dent, or a 100–400 cent bend. Re-anchor whenever the tip changes, since
`S` scales roughly as `(a/w)²`.

**Verify measurements before believing them.** A Butterworth band-t60 measurement over-reads
by 1.7–2.0×, because filter skirts let the louder low partials dominate the tail. Test the
measurement on a signal whose decay you set yourself before tuning anything against it.

**Check the whole family, not the one member you can solve.** The piston calibrates the air
integral for `m = 0` and says nothing about `m ≥ 1`. The factor of two it missed survived
every earlier check. The infinite-plane limit exists for every order; use limits that do.

**A contact must only take energy out.** Newton restitution applied across many simultaneous
contacts is not guaranteed passive, and here it was not: with every loss switched on, the
sheet settled onto a plateau of constant energy and rattled forever. It also produced a
+34 dB high-band tail that sounded exactly like a snare. Audit energy with the losses off;
`test_wires_are_passive` does.

**Do not merge what decorrelates.** Strands 3.5 mm apart are closer than the sheet's modes
can resolve, so bundling them looked free. It is not: a bundle forces its strands to hit in
lockstep, and a heavy lockstep bundle clamps the sheet. The high-band excess went +4, +21,
+32, +35 dB at 4, 8, 10 and 20 (unbundled) and did not converge until the bundling was gone.

**Crest factor is not loudness.** Raw modal pressure has a crest of 50–120; a real
close-miked drum recording has 8–15. Peak-normalising leaves the body 30 dB down. The gap is
the preamp and the limiter — a recording chain, not the instrument. Label it as such.

---

## Reference numbers

Hold these; they catch most regressions.

```
13" tom, a = 0.164, c = 100 m/s, σ_μ = 0.35306, f_max = 5 kHz
  327 distinct modes, 638 with degeneracy
  f: 233.3780  371.85  498.39  535.70  619.17  680.83  736.42  816.86 Hz
  c = 100.000000000000    f1 = 233.377972    beta = 1702.8548343087
  S.max = 1.1473048982162519e-04
  gamma_c = 1.0933294879565632    gamma_a = 1.0900814082364811
  hold gap = 5.15 cents

Air: A·j at j > 80 (plane limit 1)      m = 0: 1.02      m >= 1: 1.01
Timpano preset through the full kettle, nothing fitted
  partial ratios 1 : 1.523 : 2.028 : 2.522 : 3.010   (measured 1 : 1.5 : 1.99 : 2.44 : 2.89)

Cavity, a = 0.164, T = 3471, 10 mil; second sheet x0.85, 7.5 mil; 34 cm
lowest coupled modes involving a sheet, Hz
                           air off (= the old rank-one model)
  open back                231.4  531.7  834.9  1140.1  1447.8  1758.5
  sealed, one sheet        271.6  535.7  835.9  1140.5  1448.0  1758.6
  two sheets, sealed       237.4  321.9  535.6   835.9  1140.5  1448.0
                           air on, modal cavity
  open back                196.0  502.7  806.1  1110.2  1418.5  1729.7
  sealed, one sheet        206.3  304.1  434.4   457.9   555.7   556.3
  two sheets, sealed       167.9  248.0  303.2   308.2   434.4   454.6

Snare preset, 20 strands, 30 N, 1.6 g/m, 0.5 mm bed
  2-16 kHz vs no wires, 60-150 ms: +4.5 dB    strands settle: 130 ms

Piston calibration (do this before trusting any impedance code)
  ∫₀^∞ J₁(u)²/u² du = 4/(3π) = 0.424413      M_add = 8ρa³/3
```

---

## Layout

```
build_tables.py          run once; writes the two generated tables
render.py                render presets to .wav
index.html               root redirect to the instrument, for GitHub Pages
.nojekyll                serve the files as they are; see Putting it online

physicaldrum/
  constants.py           physical and engine constants, shared with the JS
  materials.py           rho, E, nu, eta for eight sheet materials
  tables.py              the mode table and air table: generation and loading
  air.py                 added-mass integral; self-consistent loaded density
  radiation.py           Rayleigh far field, including coincidence
  sheet.py               one clamped sheet: mode selection, frequencies, damping
  cavity.py              the cavity's modes: symmetric coupled blocks, closed-form added mass
  wires.py               snare strands: geometry, contact solve
  engine.py              build() and render()
  presets.py             starting points, not identities

tests/
  test_parity.py         Python vs the browser engine
  test_physics.py        vs exact results and published measurements
  golden_js.json         reference values extracted from the JS

tools/
  extract_js_golden.py   refresh golden_js.json (needs node)
  embed_tables.py        write the air table and cavity zeros into the page

web/
  physicalDrumBeta.html  the browser engine, self-contained
```

`modes_full.json` and `air_table.npz` are generated and gitignored — they are
pure functions of `build_tables.py`, and the JSON is 1.1 MB that would churn on
every change to the mode-selection rule. The page carries its own copies inline;
after regenerating, run `python tools/embed_tables.py` and then
`python tools/extract_js_golden.py`.

### Source documents

Three documents predate all of this and are canonical for the physics. They live
outside this repo, in the project's `docs/`:

- **`drum-physics-and-simulation.md`** — the physics/method primer. §2 ideal membrane, §3.1
  air loading, §3.3 damping, §5 excitation, §6 radiation, §8 modal synthesis, §10
  energy-conserving schemes, §13.2 the build order.
- **`implementation-guide.md`** — the canonical build plan. Part 1 is Stages 1–8, each with
  measured acceptance numbers. Appendix A has parameter tables for three real drums;
  Appendix B is the debugging playbook.
- **`physics-intuition.md`** — intuition-first companion, structured *Think it → Hear it →
  Check it → Problems*. Hints at the back, no answers, by design.

---

## The browser build

`web/physicalDrumBeta.html` is a single page carrying the whole engine in JavaScript, with
the mode table, the air table and the cavity's zeros inlined (which is most of its 362 KB). It exists to make the model **audible while you
change it**, and to serve as an oracle: it renders the same physics the Python engine does,
so a disagreement between them is a bug in one of them.

Six slots, renameable, each an independent parameter set. Settings and names persist in that
browser only (`localStorage`, under a key carrying the engine version so an engine change
invalidates them rather than restoring numbers that no longer mean the same thing);
**Reset** wipes it and restores the defaults.

Presets — bass drum, timpano, steel sheet, latex sheet, kevlar disc and others — are places
to start, not identities.

Building a slot takes 2–150 ms. Rendering a strike takes ~0.1 s — except with wires,
~1.5 s, most of it the contact solve over 120 strand points against 623 sheet modes for the
first 130 ms. So a slider drag updates the build-side readouts as it moves and renders once
it has been still for a quarter of a second. The whole kit builds in ~5 s.

### Putting it online

The page is static, so GitHub Pages can serve it as-is — no build, no action, no
dependencies:

**Settings → Pages → Source: Deploy from a branch → Branch: `main` → Folder: `/ (root)` → Save.**

A minute later the repo is live at `https://<user>.github.io/physicalDrumBeta/`, and
that bare URL is the instrument: `index.html` at the root is a redirect to
`web/physicalDrumBeta.html`. Pages serves static files only, so there is no
server-side redirect to use; a meta refresh is the standard substitute, and the
visible button on that page is the fallback for anyone whose browser suppresses it.

`.nojekyll` is there because Pages runs Jekyll by default, and Jekyll quietly drops
any path beginning with an underscore. Nothing here has one today; the file makes sure
nothing added later disappears from the site without a word.

The published page loads the same as a local double-click. It still asks
`fonts.googleapis.com` for a stylesheet and still falls back to system fonts if
that is blocked; nothing else leaves the browser, and no audio is generated
anywhere but on the visitor's own machine.

---

## Not modelled

Named plainly, because a model is only useful if you know where it stops.

- **How hard a real stroke is.** The snare wires buzz for ~40 ms at the preset strike and
  for ~160 ms at ten times it — which is what a snare sounds like — but ten times it drives
  the tension model to its clamp. Either the presets' impulses are well below a real
  stroke's, or the tension model overstates the stretch of a hard hit, or both. This is the
  most important open question the wires raise, and it is a calibration question, not a
  rendering one.
- **Snare strands as coils.** Each strand is a string. A real strand is a coiled spring, with
  its own high-frequency modes and turns that collide with each other.
- **Shell flexure**, the room, and lug asymmetry. The shell is also a second path from the
  struck sheet to the far one, through the bearing edges; here only the air connects them.
- **The cavity's shape.** It is a rigid cylinder. A kettle is not.
- **Bending mode shapes.** Frequencies carry the `k⁴` term; the shapes are still the membrane
  ones.
- **von Kármán coupling.** A stiffness-dominated plate does not respond to a strike with a
  uniform tension rise — it cascades energy between modes. That is what a gong does, and it
  is the honest next boundary for plates.
