"""
Checks against things outside this codebase.

Parity says the two engines agree with each other, which they could do while
both being wrong.  These say the physics agrees with an exact result, a
published measurement, or a conservation law.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physicaldrum import Params, PRESETS, build, render                # noqa: E402
from physicaldrum import engine as E                                  # noqa: E402
from physicaldrum.air import added_mass_coeff                         # noqa: E402
from physicaldrum.constants import RHO_AIR, C_AIR                     # noqa: E402
from physicaldrum.sheet import make_sheet                             # noqa: E402
from physicaldrum.cavity import (secular, split_ties, block, gen_eigh,  # noqa: E402
                                 neumann_zeros, overlap, rad_norm)
from physicaldrum.wires import solve_contact                          # noqa: E402
from physicaldrum import tables                                       # noqa: E402
from scipy.special import jv                                          # noqa: E402
from scipy.integrate import trapezoid                                 # noqa: E402

SR = 44100


# ------------------------------------------------------------------ the air
def test_piston_calibration():
    """The added-mass constant, against the one case with an exact answer.

    A rigid baffled piston has M_add = 8 rho a^3/3.  In the Hankel form used by
    air.py that is the integral INT J_1(u)^2/u^2 du = 4/(3 pi).  If this drifts,
    every added mass in the model is wrong by the same factor, invisibly.
    """
    u = np.linspace(1e-9, 600, 600_000)
    got = trapezoid((jv(1, u) / u) ** 2, u)
    assert abs(got - 4 / (3 * math.pi)) < 1e-6, \
        "piston integral %.8f, exact %.8f" % (got, 4 / (3 * math.pi))


def test_air_layer_is_one_over_k_for_every_order():
    """The infinite-plane limit, for every angular order -- not just m = 0.

    Far above the lowest modes a pattern of wavenumber k carries an air layer
    rho/k thick on each open face, whatever its angular order, so A j -> 1.
    The piston only checks m = 0.  This is the test that caught A being exactly
    twice too heavy for every m >= 1 (the angular factor divided in twice).
    """
    tb = tables.modes()
    rs, A = tables.air_grid()
    for sel, what in (((tb['J'] > 80) & (tb['M'] == 0), 'm = 0'),
                      ((tb['J'] > 80) & (tb['M'] >= 1), 'm >= 1')):
        p = (A[0, sel] * tb['J'][sel]).mean()
        assert abs(p - 1.0) < 0.04, "%s: A j = %.3f, the plane limit is 1" % (what, p)


def test_added_mass_asymptote():
    """Above j ~ 60, A j depends only on m/j (tight to 2% within a band)."""
    tb = tables.modes()
    sel = tb['J'] > 60
    m, j = tb['M'][sel], tb['J'][sel]
    A = added_mass_coeff(m, j, np.full(len(j), 0.30))
    u = m / j
    for lo in (0.10, 0.20, 0.30, 0.40):
        band = (u >= lo) & (u < lo + 0.10)
        if band.sum() < 20:
            continue
        p = A[band] * j[band]
        assert p.std() / p.mean() < 0.02, \
            "A*j should depend only on m/j; at m/j~%.2f the spread is %.1f%%" % (
                lo, 100 * p.std() / p.mean())


# --------------------------------------------------------------- the cavity
def test_cavity_reduces_to_rank_one():
    """Keep only the uniform cavity mode and the old model must come back exactly.

    The modal cavity's q = s = 0 term is the lumped compliance, whose exact
    eigenvalues are the roots of a secular equation.  Same answer, two routes.
    """
    a, L = 0.164, 0.34
    B = make_sheet(a, 3471, 10.0, 1.2, 16, 1.05, 0.0035, 16000, 0, True, sides=1)
    R = make_sheet(a, 3471 * 0.85, 7.5, 1.56, 20, 1.05, 0.0035, 16000, 0, True, sides=1)
    heads = []
    for S, face in ((B, 1), (R, -1)):
        for i in np.flatnonzero(S.m == 0):
            heads.append(dict(j=S.j[i], jp=S.jp[i], face=face, M=S.mass[i],
                              K=S.mass[i] * S.om[i] ** 2, R=S.R[i]))
    K, M, kinds, _, _ = block(0, heads, a, L, f_dyn=0.0, chi_pad=30.0, static=False)
    lam, _ = gen_eigh(K, M)
    om2 = np.array([h['K'] / h['M'] for h in heads])
    w = np.array([h['R'] / math.sqrt(h['M']) for h in heads])
    o = np.argsort(om2)
    ref = secular(split_ties(om2[o]), (w * w)[o], RHO_AIR * C_AIR ** 2 / (math.pi * a * a * L))
    err = np.abs(np.sqrt(lam) - np.sqrt(ref)) / np.sqrt(ref)
    assert err.max() < 1e-6, "modal cavity vs rank-one: %.2e" % err.max()


def test_cavity_side_is_a_half_space_for_fine_patterns():
    """Deep in a cavity, a fine pattern's air is the same as open air's.

    The cavity's added mass for a sheet mode, summed in closed form over every
    cavity mode, must match the same disc's added mass in a baffled half-space
    (air.py, at small r) once the pattern decays well within the depth.  Two
    independent calculations -- a Hankel integral and a sum over cylinder
    modes -- of one physical quantity.  They agree to 2-7.5%, and the cylinder
    is always the heavier: its rigid wall stops air escaping over the rim the
    way it can over a baffle, and confinement can only add mass.
    """
    a, L = 0.164, 1.0
    tb = tables.modes()
    idx = np.flatnonzero((tb['M'] >= 1) & (tb['J'] > 15) & (tb['J'] < 25))
    for i in idx[::3]:
        m, j, jp = int(tb['M'][i]), tb['J'][i], tb['JP'][i]
        ang = math.pi
        tot = 0.0
        for chi in neumann_zeros(m, j + 60):
            g = overlap(m, j, jp, chi, a)
            tot += RHO_AIR * g * g / (ang * rad_norm(m, chi, a) * (chi / a)) / math.tanh(chi / a * L)
        N = ang * (a * a / 2) * jp * jp
        half = RHO_AIR * a * added_mass_coeff([m], [j], [0.04])[0] * N
        assert 1.0 <= tot / half < 1.08, "m=%d j=%.1f: cavity %.3f of the half-space" % (m, j, tot / half)


def test_timpani_partial_ratios():
    """The whole air model, against Rossing's measured timpani partials.

    Built as a kettle: one sheet, sealed cavity, air on the open face and the
    cavity's own modes on the other.  Nothing is fitted.  An ideal membrane is
    21% out; this should be under 5%.
    """
    d = build(Params(**PRESETS['timpano']))
    m, f, pb = d.info['c_m'], d.info['c_f'], d.info['c_partB']
    fs = np.array([f[(m == mm) & (pb > 0.5)].min() for mm in (1, 2, 3, 4, 5)])
    got = fs / fs[0]
    measured = np.array([1.0, 1.5, 1.99, 2.44, 2.89])
    err = np.abs(got - measured) / measured
    assert err.max() < 0.05, \
        "timpani ratios %s vs measured %s (%.1f%% worst)" % (
            np.round(got, 3), measured, 100 * err.max())


def test_cavity_splits_the_breathing_mode():
    """Two sheets and sealed air: the lowest breathing mode becomes a pair."""
    open_back = build(Params(a=0.164, T=3471, d=3000, Tr=0))
    sealed = build(Params(a=0.164, T=3471, d=140, Tr=0.85))
    assert sealed.info['two_sheet']
    lo, hi = sealed.info['f_air']
    assert hi > lo > 0, "no air pair found"
    assert hi / lo > 1.2, "pair barely split: %.1f / %.1f Hz" % (lo, hi)
    assert lo > open_back.f1 * 0.7, "sealing should not lower the pitch that much"


def test_cavity_reaches_the_far_sheet_at_every_order():
    """The point of the modal cavity: m >= 1 patterns now cross the drum.

    The lumped model coupled only m = 0 (INT INT Phi dA = 0 otherwise).  Here,
    for m = 1..3 on the snare, some eigenmode must share its energy between the
    two sheets -- at least 10% on each.
    """
    d = build(Params(**PRESETS['snare']))
    m, pb = d.info['c_m'], d.info['c_partB']
    for mm in (1, 2, 3):
        shared = (m == mm) & (pb > 0.1) & (pb < 0.9)
        assert shared.any(), "m = %d: no eigenmode shared between the sheets" % mm


def test_secular_interlacing():
    """Rank-one reference: every root sits in its own gap."""
    rng = np.random.default_rng(0)
    om2 = np.sort(rng.uniform(1e4, 1e8, 25))
    w2 = rng.uniform(0, 1, 25) ** 2
    lam = secular(om2, w2, 5e6)
    assert np.all(lam[:-1] <= om2[1:] + 1e-6), "a root escaped its gap"
    assert np.all(lam >= om2 - 1e-6), "a root fell below its gap"


# ---------------------------------------------------------------- the wires
def test_contact_solve_is_exact():
    """Random complementarity problems: F >= 0, no overlap, no force when clear."""
    rng = np.random.default_rng(1)
    for _ in range(40):
        n = 30
        Q = rng.normal(size=(n, n))
        A = Q @ Q.T / n + np.eye(n) * 0.1
        eta = rng.normal(size=n)
        F, act = solve_contact(A, eta)
        g = eta - A @ F
        assert F.min() >= 0
        assert g.max() < 1e-10, "overlap left: %.2e" % g.max()
        assert np.abs(F * g).max() < 1e-10, "force where clear"


def test_wires_at_rest_are_silent():
    """With the preload on and no strike, nothing moves and nothing sounds.

    The strands start pressed on the sheet at the integrator's exact fixed
    point; starting from zero would fire the preload as a step.  At rest the
    output must be 100 dB under a real strike and the bed forces steady.
    """
    pr = dict(PRESETS['snare'])
    d = build(Params(**pr))
    _, r = render(d, impulse=0.0, dur=0.05, limiter=False, trace=True)
    _, rs = render(d, impulse=pr['P'] * 1e-3, dur=0.05, limiter=False, trace=True)
    quiet = np.abs(r['raw']).max() / np.abs(rs['raw']).max()
    assert quiet < 1e-5, "rest is only %.0f dB under a strike" % (20 * math.log10(quiet))
    F = r['trace']['F'][1:]
    assert F.std() < 1e-5 * F.mean(), "the preload is not steady at rest"


def test_wires_are_passive():
    """With every loss switched off, the sheet-plus-strands energy never rises.

    A contact can only push, so it can only take energy out.  Newton restitution
    on many simultaneous contacts broke that: it put energy in, the sheet
    rattled forever, and the leak sounded exactly like a snare.  This is the
    test that would have caught it.
    """
    pr = dict(PRESETS['snare'])
    saved = E.WIRE_SIG
    E.WIRE_SIG = 0.0
    try:
        def lossless():
            d = build(Params(**pr))
            d.sig[:] = 0.0
            d.beta = 0.0
            d.life[:] = 1e9
            return d
        _, r0 = render(lossless(), impulse=0.0, dur=0.003, trace=True, limiter=False)
        base = r0['trace']['E'][1]
        _, r = render(lossless(), impulse=pr['P'] * 1e-3, dur=0.25, trace=True, limiter=False)
    finally:
        E.WIRE_SIG = saved
    En = r['trace']['E'][1:] - base
    En = En[En != -base]            # samples before any wire step
    assert En.max() <= En[0] * 1.002, \
        "energy rose to %.4f of the strike's" % (En.max() / En[0])


def test_wires_rattle_and_settle():
    """A struck snare: the strands leave the sheet, hit it, and settle.

    And the rattle is heard: 60-150 ms after the strike the high band is louder
    with wires than without, when the sheet's own highs have died.
    """
    pr = dict(PRESETS['snare'])
    d = build(Params(**pr))
    y, r = render(d, impulse=pr['P'] * 1e-3, dur=0.4, limiter=False, trace=True)
    sep = r['trace']['sep']
    assert sep[: int(0.02 * SR)].max() > 0.2, "the strands never left the sheet"
    assert r['t_frozen'] < 0.4, "the strands never settled"
    d0 = build(Params(**{**pr, 'wN': 0}))
    _, r0 = render(d0, impulse=pr['P'] * 1e-3, dur=0.4, limiter=False, trace=True)

    def hi(x):
        s = x[int(0.06 * SR): int(0.15 * SR)]
        X = np.abs(np.fft.rfft(s)) ** 2
        f = np.fft.rfftfreq(len(s), 1 / SR)
        return X[(f > 2000) & (f < 16000)].sum()
    gain = 10 * math.log10(hi(r['raw']) / hi(r0['raw']))
    assert gain > 2.0, "wires add only %.1f dB to the 60-150 ms highs" % gain


# -------------------------------------------------------------- the sheet
def test_stiffness_dominates_a_metal_sheet():
    """A steel foil must be bending-dominated; a Mylar head must not be."""
    steel = build(Params(mat='Steel', a=0.300, T=9000, h=6.0, fmax=16000, d=3000, Tr=0))
    mylar = build(Params(mat='Mylar', a=0.164, T=3471, h=10.0, fmax=16000, d=3000, Tr=0))
    assert steel.info['stiffness'] > 0.5, \
        "steel stiffness share only %.2f" % steel.info['stiffness']
    assert mylar.info['stiffness'] < steel.info['stiffness']


def test_tension_bend_is_monotone_in_impulse():
    """Hit it harder, it bends further -- with no discontinuity."""
    d = build(Params(mat='Aluminium', a=0.120, T=26000, h=10.0, d=3000, Tr=0))
    bends = [render(d, impulse=P * 1e-3)[1]['bend']
             for P in (2, 5, 10, 20, 40)]
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(bends, bends[1:])), \
        "bend not monotone: %s" % np.round(bends, 1)
    jumps = [b2 / max(b1, 1e-3) for b1, b2 in zip(bends, bends[1:])]
    assert max(jumps) < 12, "discontinuous jump in bend: %s" % np.round(bends, 1)


def test_render_is_finite_across_the_parameter_space():
    """No NaN anywhere a user can reach."""
    for mat in range(8):
        for a in (0.05, 0.30):
            for T in (200, 20000):
                for h in (1.0, 30.0):
                    d = build(Params(mat=mat, a=a, T=T, h=h, d=3000, Tr=0,
                                     fmax=14000))
                    y, r = render(d, impulse=1e-2, dur=0.2)
                    assert np.all(np.isfinite(y)), \
                        "non-finite audio at mat=%d a=%.2f T=%d h=%.1f" % (mat, a, T, h)
                    assert np.isfinite(d.f1)
    for depth in (30, 140, 600):
        for wN in (0, 8, 30):
            d = build(Params(**{**PRESETS['snare'], 'd': depth, 'wN': wN}))
            y, r = render(d, impulse=7e-3, dur=0.1)
            assert np.all(np.isfinite(y)), "non-finite audio at d=%d wN=%d" % (depth, wN)


if __name__ == '__main__':
    import time
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            t = time.time()
            fn()
            print("ok  %-48s %5.1fs" % (name, time.time() - t))
