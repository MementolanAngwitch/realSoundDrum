# physicalDrumBeta

A struck sheet, solved from its physical properties. No samples, no fitted envelopes,
no oscillator banks tuned by ear. We design a clamped circular sheet that is given a material, a size, a tension and a thickness, air is put behind it, and something with its own physical properties hits it.

---

## Quick start

```bash
pip install -r requirements.txt     # numpy, scipy -- nothing else
python build_tables.py              # generates the two tables (~1 min)
python render.py                    # renders every preset to .wav
python tests/test_parity.py         # Python vs the browser engine
python tests/test_physics.py        # against exact results and measurements
```

Python 3.7 or newer, on any OS. `build_tables.py` must run once before anything else. Everything after that is offline and self-contained.

For the browser engine, open `web/physicalDrumBeta.html` The engine, the mode table and the air table are all inlined.

```python
from physicaldrum import Params, build, render, PRESETS

d = build(Params(mat='Steel', a=0.30, T=9000, h=6.0, d=3000, Tr=0))
y, r = render(d, impulse=0.028)
print(d.f1, d.info['stiffness'], r['bend'])
```

---

## The model

The chain, in the order the code applies it.

### 1. Modes

A clamped circular sheet of radius `a` has mode shapes

```
Φ_mn(r,θ) = J_m(j_mn · r/a) · cos(mθ)          j_mn = the n-th zero of J_m
```

`m ≥ 1` modes are doubly degenerate (a `cos mθ` and a `sin mθ` partner). Gauge-fixing them to the strike axis is valid for one strike point only.

Normalisation matters.

```
∬ Φ² dA = ε_m · (πa²/2) · J_m′(j_mn)²          ε_m = 2 if m = 0 else 1
```

Store `J_m′(j_mn)` signed. `sqrt` of the squared value loses the sign, and the sign is
physical as  it sets the relative phase of the partials.

### 2. Frequency: tension *and* bending stiffness

```
ω² = (T k² + D k⁴) / σ_eff        k = j_mn/a,   D = E h³ / 12(1−ν²)
```

The `k⁴` term is not a correction. On a 10 mil Mylar head it is already half the restoring
force by `j = 120`, and 98% on a steel foil:

| j   | T k²     | D k⁴     | stiffness share |
| --- | -------- | -------- | --------------- |
| 10  | 1.29e+07 | 8.99e+04 | 0.7%            |
| 80  | 8.26e+08 | 3.68e+08 | 30.8%           |
| 120 | 1.86e+09 | 1.86e+09 | 50.1%           |
| 170 | 3.73e+09 | 7.51e+09 | 66.8%           |

A model with only `T k²` can only ever be a drum. Keep the two terms in separate columnsall the way through. Tension modulation scales only the first, and the interface contract depends on it.

Caveat: the mode shapes used are the membrane ones. A stiffness-dominated sheet is
therefore approximated, not solved. The stiffness at topreadout says how far into that
approximation you are.

### 3. Air added mass

A moving sheet drags a layer of air with it. In the Hankel domain the mode's transform is the same closed form as its far field, and the reactive part is the tail beyond `u = ωa/c`:

```
added σ (one open side) = ρ_air · a · A
A = 4j²/ε · ∫₀^∞ [ J_m(u)/(u²−j²) ]² ds ,   u = √(s² + (rj)²) ,   r = c_phase/c_air
```

- The `s` substitution is not cosmetic. A naïve grid in `u` across the singularity gives the
  (1,1) mode a 1071-cent drop instead of 507.
- One open side. The other face looks into the sealed cavity, whose air is a compliance already carried by the `K_air` term, not an unbounded half-space.
- Calibrate the overall constant on the rigid baffled piston, `M = 8ρa³/3`. It reproduces to six decimal places, and it catches normalisation errors that are otherwise invisible.

This is the term that makes a drum a pitched instrument: it loads the low modes hardest, which drags the Bessel ratios toward harmonic.

Validation: tune a timpano so the loaded (1,1) lands on 110 Hz and compare partial
ratios to Rossing's measured set:

```
ideal membrane   1  1.34   1.665  1.98   2.289     21.0% max error
this model       1  1.468  1.917  2.356  2.787      3.7% max error
measured         1  1.5    1.99   2.44   2.89
```

Above `j ≈ 40`, `A → p(r)/j` — one air cell thick, which is exactly the infinite-plane
`ρ_air/k` limit. The finite-disc integral only earns its cost on the low modes.

### 4. The cavity

The enclosed air is one compliance shared by both sheets, so its generalised force is a
rank-one stiffness update:

```
K = diag(m ω²) + K_air · R Rᵀ ,   K_air = ρc²/V ,   R_i = ∬ Φ_i dA
```

Rank one means no eigensolver: the eigenvalues are the roots of
`1 + K_air · Σ w²/(ω²−λ) = 0`, and they interlace the `ω²`, one per gap, so plain bisection is unconditionally robust. Because `R_i = 0` for every `m ≥ 1`, only the breathing modes couple, roughly 13 + 13 + a vent DOF, out of hundreds.

The vent is a zero-stiffness DOF with mass `ρ_air A_p L_eff` so Helmholtz falls out for free.

Two identical sheets give a repeated root, which leaves the bisection with a zero-width
bracket. That repeated eigenvalue is real physics — the antisymmetric combination has zero net volume, so the air cannot see it — and splitting the tie by a relative `1e-7` lands on exactly that limit.

### 5. Radiation

The Rayleigh far field of a baffled disc, which for a *clamped* mode has a closed form
because `J_m(j_mn) = 0` at the rim:

```
D_mn = j · J_m′(j) · J_m(β) / (β² − j²) ,     β = (ωa/c_air) · sin θ
```

- `∬ Φ dA`   is the `ka → 0` monopole limit of this, and it is exactly zero for every `m ≥ 1`. Using it silences most of the mode set.
- At `θ = 0` the expression does vanish for `m ≥ 1`: an on-axis mic really does hear only the breathing modes. Off axis it does not.
- `β → j` is coincidence, where the sheet's phase speed reaches the speed of sound. The expression is 0/0 there and the limit is `πa² J_m′(j)²` — finite and large. That is a real radiation peak, and it is why a cymbal is loud and a drumhead is not. A steel foil reaches it at `j = 144`, inside the audio band.

### 6. Damping

```
σ_i = σ₀ + η · ω_i / 2
```

A constant loss factor Q. A polymer loses a fixed *fraction* of the strain energy per cycle, not a fixed amount per second. `η ≈ 0.015` for Mylar lands inside the measured
t60 range for a real tom in every octave from 200 Hz to 3 kHz.

`σ = σ_a + σ_b f²` is a fitted curve with no mechanism, and measured against a render it is ~3.5× too weak at 200 Hz and ~2× too strong at 3 kHz — low partials sustaining while the highs vanish, a metallic sound

### 7. The strike

An impulse `P` delivered through a Gaussian contact patch of radius `w`, over a contact time

```
τ = 2w / c              floored by the striker's own softness (felt, rubber)
```

The dent at contact spreads at the sheet's wave speed, so the tip stays down about as long as the sheet needs to get out from under it. Tip radius is the single biggest control on brightness, because it band-limits the strike twice over: through the spatial roll-off `exp(−(kw)²/2)` and through `τ`.

### 8. Tension modulation

Denting a sheet stretches it, which raises its tension, which raises every tension-borne
partial and then relaxes:

```
S = ∬ |∇u|² dA = Σ k_i² q_i² norm_i
γ = √(1 + β S)              β = E h / (2π a² T)
ω² → γ² · ω_T² + ω_D²       γ multiplies ONLY the tension term
```

The tension rise is not instantaneous. It is carried across the sheet by the in-plane
wave at `c_L = √(E/ρ)` — 1700 m/s in Mylar, 5100 in aluminium — so global tension lags local stretch by about one crossing time, `τ_L = 2a/c_L`. Treating it as instantaneous lets the first few samples, when every mode is still in phase and the modal sum for `S` is at its most coherent and least converged, set γ for the whole note. On a stiff sheet that spike alone pinned γ at its clamp: 74 cents at `P = 5 mN·s`, 814 at 7.18, nothing in between. One pole with `τ_L` removes it.

### 9. Integration

Per mode, state is (q, v), position and velocity, advanced by the exact damped-oscillator
rotation:

```
q′ = e^(−σΔt) [ q·cos(ω_dΔt) + (v + σq)/ω_d · sin(ω_dΔt) ]
v′ = e^(−σΔt) [ v·cos(ω_dΔt) − (ω²q + σv)/ω_d · sin(ω_dΔt) ]
```

The two-pole form `q[n+1] = a1·q[n] − a2·q[n−1]` is exact at a *fixed* frequency and is what to use when there is no modulation. But swapping `a1` while keeping both past samples is not energy-consistent: the pair encodes an amplitude that means something different under the new frequency. Position and velocity are physical, so the same rotation with a new ω is simply the same state seen by a different oscillator. The strike is then just a velocity kick.

---

## Things to fix

`∬ Φ dA` is a limit, not a definition. It is the `ka → 0` monopole term. Using it as the
radiation weight silences every `m ≥ 1` mode — 187 of 200 on a typical head.

Damping laws need a mechanism. If you cannot say what physical process a damping term represents, it is a curve fit, and it will be wrong somewhere in the band.

Band and ceilings are invisible. A mode table that stops at `j < 82` caps a 16" tom at
7.9 kHz. A real close-miked drum has usable energy to 12–16 kHz. Nothing errors; the sound is just dull.

Prune by radiation, not by frequency. Reaching 16 kHz naively needs ~2300 modes, but
`J_m(β)` is exponentially small once `m ≫ β`, so high angular orders are acoustically
invisible — only ~290 of those 2300 clear −50 dB. Keep those plus everything below a low `j` cut (quiet, but they carry `S`), and the band triples for ~300 extra modes.

Order modes by lifetime. With constant Q a 10 kHz mode is inaudible after ~25 ms whilethe fundamental runs for a second. Sorting by death sample and shrinking an active count is~15× and it is exact, not an approximation.

`S` only converges once the band resolves the striker tip. A point force on a membrane
has a 1/r slope, so `∬|∇u|² dA` diverges logarithmically; the tip radius regularises it, but
only if modes out to `j ~ 3a/w` are present. Measured on a 16" tom:

| tip     | 200 modes | 500    | 1000   | 1500   | spread  |
| ------- | --------- | ------ | ------ | ------ | ------- |
| 10.0 mm | 1.0220    | 1.0220 | 1.0220 | 1.0220 | 0%      |
| 3.5 mm  | 1.1230    | 1.1484 | 1.1588 | 1.1588 | 3%      |
| 1.5 mm  | 1.1660    | 1.2311 | 1.3973 | 2.0000 | **72%** |

Below that, the pitch bend is a truncation artifact. Floor the tip at what the retained band.

γ can push modes past Nyquist. Once the band reaches 16 kHz, a γ of only 1.38 folds the top modes to arbitrary low frequencies — an audible screech. Clamping them *to* Nyquist is no better: hundreds then pile onto one frequency and ring in unison. Simply do not modulate modes that could ever exceed it; they are above 13.8 kHz, where a few hundred cents of glide is inaudible.

Impulse is the wrong thing to hold fixed.  Anchor calibration on something physical and
measurable, a 4 mm dent, or a 100–400 cent bend. Re-anchor whenever the tip changes, since `S` scales roughly as `(a/w)²`.

Verify measurements before believing them. A Butterworth band-t60 measurement over-reads by 1.7–2.0×, because filter skirts let the louder low partials dominate the tail. Test the measurement on a signal whose decay you set yourself before tuning anything against it.

Crest factor is not loudness. Raw modal pressure has a crest of 50–120; a real
close-miked drum recording has 8–15. Peak-normalising leaves the body 30 dB down. The gap is the preamp and the limiter which is a recording chain, not the instrument. 

---

## Reference numbers

```
13" tom, a = 0.164, c = 100 m/s, σ_μ = 0.35306, f_max = 5 kHz
  327 distinct modes, 638 with degeneracy
  f: 233.3780  371.85  498.39  535.70  619.17  680.83  736.42  816.86 Hz
  c = 100.000000000000    f1 = 233.377972    beta = 1702.8548343087
  S.max = 1.1473048982162519e-04
  gamma_c = 1.0933294879565632    gamma_a = 1.0900814082364811
  hold gap = 5.15 cents

Air loading, one open side, validated on a timpano (a = 0.33, 7.5 mil, f11 = 110 Hz)
  partial ratios 1 : 1.468 : 1.917 : 2.356 : 2.787   (measured 1 : 1.5 : 1.99 : 2.44 : 2.89)
  added mass on (0,1) of a 16" tom: ~40% of the head's own mass

Cavity, 16" tom, 14 cm deep
  bare sheet              233.4  535.7  839.8  1144.3  1449.0  1753.7
  sealed, one sheet       319.0  546.6  842.4  1145.3  1449.4  1754.0
  two sheets, sealed      241.4  400.2  545.8   600.7   842.6   912.3
  two sheets + 12 mm vent  90.9  241.5  417.5   546.9   603.7   842.8

Piston calibration (do this before trusting any impedance code)
  ∫₀^∞ J₁(u)²/u² du = 4/(3π) = 0.424413      M_add = 8ρa³/3
```

---

## Layout

```
build_tables.py          run once; writes the two generated tables
render.py                render presets to .wav
index.html               root redirect to the instrument, for GitHub Pages
.nojekyll                keeps Pages from dropping underscore-prefixed paths

physicaldrum/
  constants.py           physical and engine constants, shared with the JS
  materials.py           rho, E, nu, eta for eight sheet materials
  tables.py              the mode table and air table: generation and loading
  air.py                 added-mass integral; self-consistent loaded density
  radiation.py           Rayleigh far field, including coincidence
  sheet.py               one clamped sheet: mode selection, frequencies, damping
  cavity.py              rank-one stiffness update, secular solve
  engine.py              build() and render()
  presets.py             starting points, not identities

tests/
  test_parity.py         Python vs the browser engine
  test_physics.py        vs exact results and published measurements
  golden_js.json         reference values extracted from the JS

tools/
  extract_js_golden.py   refresh golden_js.json (needs node)

experiments/             not wired in; see experiments/README.md
web/
  physicalDrumBeta.html  the browser engine, self-contained
```

`modes_full.json` and `air_table.npz` are generated and gitignored — they are
pure functions of `build_tables.py`, and the JSON is 1.1 MB that would churn on
every change to the mode-selection rule.

---

## Not modelled

- Snare wires. The one-sided contact model exists and works, but the cavity is treated as a single uniform pressure, which cannot drive the far sheet at any angular order above zero. The wires rattle against one slow breathing mode rather than a real surface. The fix is the cavity's actual mode set (see below), and until then the control is absent
  rather than inert.
- The cavity's non-uniform modes. Above `1.8412·c/(2πa)` — 565 Hz for a 14" drum the pressure inside is not uniform, and the sheets couple at every angular order, not just `m = 0`. The overlap integral is Lommel again, the same closed form as the far field and the added mass, and the current rank-one model is its `q = s = 0` term. Half built; the coupled solve validates against rank-one exactly, but it is a non-symmetric generalized eigenproblem and its eigenvectors are not yet trustworthy.
- Shell flexure, the room, and lug asymmetry.
- Bending mode shapes. Frequencies carry the `k⁴` term; the shapes are still the membrane ones.
- von Kármán coupling. A stiffness-dominated plate does not respond to a strike with a uniform tension rises, it cascades energy between modes. That is what a gong does, and it is the honest next boundary for plates.
