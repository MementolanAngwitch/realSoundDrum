"""
Checks against things outside this codebase.

Parity says the two engines agree with each other, which they could do while
both being wrong.  These say the physics agrees with an exact result or a
published measurement.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physicaldrum import Params, build, render                    # noqa: E402
from physicaldrum.air import added_mass_coeff                     # noqa: E402
from physicaldrum.constants import RHO_AIR                        # noqa: E402
from physicaldrum.sheet import make_sheet, MIL                    # noqa: E402
from physicaldrum.cavity import secular                           # noqa: E402
from physicaldrum import tables                                   # noqa: E402
from scipy.special import jv                                      # noqa: E402
from scipy.integrate import trapezoid                             # noqa: E402


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


def test_timpani_partial_ratios():
    """Air loading, against Rossing's measured timpani partials.

    Tune a timpano so the LOADED (1,1) lands on 110 Hz, then compare ratios.
    An ideal membrane is 21% out; this should be under 5%.
    """
    a, h = 0.330, 7.5

    def f11(T):
        s = make_sheet(a, T, h, 1.0, 15, 1.05, 0.010, 1500, mat='Calfskin')
        i = np.flatnonzero(s.m == 1)[0]
        return s.f[i]

    lo, hi = 200.0, 9000.0
    for _ in range(45):
        mid = (lo + hi) / 2
        if f11(mid) < 110:
            lo = mid
        else:
            hi = mid
    s = make_sheet(a, (lo + hi) / 2, h, 1.0, 15, 1.05, 0.010, 1500, mat='Calfskin')
    idx = [np.flatnonzero(s.m == mm)[0] for mm in (1, 2, 3, 4, 5)]
    got = s.f[idx] / s.f[idx[0]]
    measured = np.array([1.0, 1.5, 1.99, 2.44, 2.89])
    err = np.abs(got - measured) / measured
    assert err.max() < 0.05, \
        "timpani ratios %s vs measured %s (%.1f%% worst)" % (
            np.round(got, 3), measured, 100 * err.max())


def test_added_mass_asymptote():
    """Above j ~ 40 the finite-disc integral must collapse onto the plane result.

    The infinite-plane added mass is rho_air/k, i.e. A ~ p/j -- one air cell
    thick.  For a finite disc A*j is not a single constant but a function of
    m/j, tight to ~1% once m/j > 0.1.  (Below that, the low-m radial modes, it
    spreads ~15%, which is why the table keeps exact values for them rather than
    using this asymptote -- see tables.py.)
    """
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


def test_secular_interlacing():
    """Rank-one update: every root must sit in its own gap.

    That interlacing is what makes plain bisection safe here.  If a root escapes
    its bracket the cavity solve is silently wrong.
    """
    rng = np.random.default_rng(0)
    om2 = np.sort(rng.uniform(1e4, 1e8, 25))
    w2 = rng.uniform(0, 1, 25) ** 2
    lam = secular(om2, w2, 5e6)
    assert np.all(lam[:-1] <= om2[1:] + 1e-6), "a root escaped its gap"
    assert np.all(lam >= om2 - 1e-6), "a root fell below its gap"


def test_cavity_splits_the_breathing_mode():
    """Two sheets and sealed air: the lowest breathing mode becomes a pair.

    Heads moving together, then against each other.  This is the single most
    drum-like thing the cavity does, and it is easy to lose.
    """
    open_back = build(Params(a=0.164, T=3471, d=3000, Tr=0))
    sealed = build(Params(a=0.164, T=3471, d=140, Tr=0.85))
    assert sealed.info['two_sheet']
    lo, hi = sealed.info['f_air']
    assert hi > lo > 0, "no air pair found"
    assert hi / lo > 1.2, "pair barely split: %.1f / %.1f Hz" % (lo, hi)
    assert lo > open_back.f1 * 0.8, "sealing should not lower the pitch much"


def test_stiffness_dominates_a_metal_sheet():
    """A steel foil must be bending-dominated; a Mylar head must not be."""
    steel = build(Params(mat='Steel', a=0.300, T=9000, h=6.0, fmax=16000, d=3000, Tr=0))
    mylar = build(Params(mat='Mylar', a=0.164, T=3471, h=10.0, fmax=16000, d=3000, Tr=0))
    assert steel.info['stiffness'] > 0.5, \
        "steel stiffness share only %.2f" % steel.info['stiffness']
    assert mylar.info['stiffness'] < steel.info['stiffness']


def test_tension_bend_is_monotone_in_impulse():
    """Hit it harder, it bends further -- with no discontinuity.

    Before the in-plane lag was added, a stiff sheet jumped from 74 cents to the
    clamp between two neighbouring impulses, because the first few samples (when
    every mode is still in phase) set gamma for the whole note.
    """
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


if __name__ == '__main__':
    import time
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            t = time.time()
            fn()
            print("ok  %-42s %5.1fs" % (name, time.time() - t))
