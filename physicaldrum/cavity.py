"""
The air sealed behind the sheet.

Below the first non-uniform cavity mode the enclosed air is a single compliance
shared by every surface, so its generalised force is a RANK-ONE stiffness update:

    K = diag(m omega^2) + K_air R R^T ,   K_air = rho c^2/V ,   R_i = INT INT Phi_i dA

Rank one means no eigensolver is needed.  The eigenvalues are the roots of

    1 + K_air SUM w_i^2/(omega_i^2 - lambda) = 0 ,   w_i = R_i/sqrt(m_i)

and they INTERLACE the omega_i^2 — exactly one root in each gap — so plain
bisection is unconditionally robust and needs no starting guess.

Because R_i = 0 for every m >= 1, only the breathing modes couple: roughly
13 + 13 + a vent, out of hundreds.  Everything else is untouched, which is what
makes this cheap.

The vent is a zero-stiffness degree of freedom with mass rho_air A_p L_eff; the
Helmholtz resonance falls out for free.

TRAP: two identical sheets give a repeated root, leaving the bisection with a
zero-width bracket.  That repeated eigenvalue is real physics — the antisymmetric
combination has zero net volume, so the air cannot see it — and splitting the tie
by a relative 1e-7 lands on exactly that limit.

WHERE THIS STOPS BEING TRUE: above 1.8412 c/(2 pi a) — 565 Hz for a 14" drum —
the pressure inside is no longer uniform, and the two sheets couple at every
angular order rather than only m = 0.  See experiments/cavity_modes.py.
"""
import numpy as np

_TIE = 1e-7          # relative split for repeated roots
_BISECT = 110


def split_ties(om2):
    """Nudge exact duplicates apart so every bracket has width."""
    out = np.array(om2, dtype=float)
    for t in range(1, len(out)):
        lo = out[t - 1] * (1 + _TIE) + 1e-6
        if out[t] < lo:
            out[t] = lo
    return out


def secular(om2, w2, K):
    """Eigenvalues of diag(om2) + K w w^T.  `om2` must be sorted ascending."""
    n = len(om2)
    lam = np.zeros(n)
    tw = float(np.sum(w2))

    def g(l):
        return 1.0 + K * np.sum(w2 / (om2 - l))

    for i in range(n):
        lo = om2[i]
        hi = om2[i + 1] if i < n - 1 else om2[-1] + K * tw + 1.0
        a = lo + 1e-9 * max(1.0, abs(lo))
        b = hi - 1e-9 * max(1.0, abs(hi))
        for _ in range(_BISECT):
            mid = 0.5 * (a + b)
            if g(mid) > 0:
                b = mid
            else:
                a = mid
        lam[i] = 0.5 * (a + b)
    return lam


def eigenvector(om2, w, lam_c):
    """Mass-normalised eigenvector for one root, unit length."""
    z = w / (om2 - lam_c)
    return z / np.linalg.norm(z)
