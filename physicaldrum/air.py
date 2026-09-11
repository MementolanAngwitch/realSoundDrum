"""
Air added mass on a clamped circular sheet.

A moving sheet drags a layer of air with it, and for a drumhead that layer is a
term rather than a correction: ~40% of the head's own mass on the lowest mode of
a 16" tom.  It loads the LOW modes hardest, which drags the Bessel ratios
1 : 1.34 : 1.67 : 1.98 toward harmonic — and that pull is why a drum reads as
pitched at all rather than as a struck plate.

The mode's Hankel transform is the same closed form as its far field (see
`radiation.py`), and the reactive part of the baffled-plane impedance is the
tail beyond u = omega a/c_air:

    added surface density (one open side) = rho_air * a * A
    A = 4 j^2/eps  INT_0^inf [ J_m(u)/(u^2 - j^2) ]^2 ds ,  u = sqrt(s^2 + (r j)^2)

with r = c_phase/c_air.

Two things that are easy to get wrong and cost a factor of two each:

* THE SUBSTITUTION.  s = sqrt(u^2 - u_a^2) turns u du/sqrt(u^2 - u_a^2) into ds
  and removes the singularity exactly.  A naive grid in u starting just above
  u_a gives the (1,1) mode a 1071-cent drop where the truth is 507.

* ONE open side, not two.  The other face looks into the sealed cavity, whose
  air is a compliance already carried by the cavity stiffness, not an unbounded
  half space.

CALIBRATION.  The overall constant 2 pi rho a^3 is fixed on the one case with an
exact answer — the rigid baffled piston, M = 8 rho a^3/3 — which it reproduces to
six decimal places.  Do this before trusting any new impedance code; it catches
normalisation errors that are otherwise invisible.

VALIDATION.  Tune a timpano so the loaded (1,1) lands on 110 Hz and compare
partial ratios to Rossing's measured set:

    ideal membrane   1  1.34   1.665  1.98   2.289     21.0% max error
    this model       1  1.468  1.917  2.356  2.787      3.7% max error
    measured         1  1.5    1.99   2.44   2.89

Above j ~ 40, A -> p(r)/j — one air cell thick, which is exactly the
infinite-plane rho_air/k limit.  The finite-disc integral only earns its cost on
the low modes.
"""
import numpy as np
from scipy.special import jv
# scipy's trapezoid, not numpy's: np.trapezoid exists only on numpy >= 2.0 and
# np.trapz was removed in it, so either name alone pins the numpy major version.
# scipy has carried this one since 1.6 and it works on both.
from scipy.integrate import trapezoid

from .constants import RHO_AIR, C_AIR


def added_mass_coeff(m, j, r, n_quad=1400):
    """Dimensionless A for arrays of (m, j, r).  Slow — use tables.air_lookup."""
    m = np.asarray(m)
    j = np.asarray(j, dtype=float)
    r = np.asarray(r, dtype=float)
    ua = r * j
    smax = np.maximum(80.0, 14 * j)
    t = np.linspace(0.0, 1.0, n_quad)[None, :]
    s = t * smax[:, None]
    u = np.sqrt(ua[:, None] ** 2 + s ** 2)
    JM = np.empty_like(u)
    for mm in np.unique(m):                       # jv wants a scalar order
        sel = (m == mm)
        JM[sel] = jv(int(mm), u[sel])
    f = (JM / (u * u - j[:, None] ** 2)) ** 2
    Q = trapezoid(f, s, axis=1)
    eps = np.where(m == 0, 2.0, 1.0)
    return 4 * j * j * Q / eps


def loaded_density(k, idx, T, D, smu, a, sides=1, iters=14, relax=0.65):
    """Self-consistent effective surface density and omega^2 split.

    omega depends on sigma_eff and sigma_eff depends on omega (through the mode's
    own phase speed), so this iterates.  Returns (sigma_eff, omT2, omD2) where

        omT2 = T k^2/sigma_eff      the part tension owns
        omD2 = D k^4/sigma_eff      the part bending owns

    kept apart because tension modulation scales only the first.
    """
    from .tables import air_lookup
    se = np.full_like(k, smu, dtype=float)
    k2 = k * k
    k4 = k2 * k2
    for _ in range(iters):
        w2 = (T * k2 + D * k4) / se
        r = np.sqrt(w2) / k / C_AIR               # this mode's own phase speed
        se = (1 - relax) * se + relax * (smu + sides * RHO_AIR * a * air_lookup(idx, r))
    return se, T * k2 / se, D * k4 / se
