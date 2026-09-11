"""
What reaches the microphone.

The Rayleigh far field of a baffled disc.  For a *clamped* mode it has a closed
form, because J_m(j_mn) = 0 at the rim kills one of Lommel's two terms:

    D_mn = j J_m'(j) J_m(beta) / (beta^2 - j^2),   beta = (omega a/c_air) sin(theta)

Three things worth knowing about it:

* INT INT Phi dA is NOT "radiation".  It is the ka -> 0 monopole limit of this
  expression, and it is exactly zero for every m >= 1.  Using it as the output
  weight silences most of the mode set — 187 of 200 on a typical head — which
  sounds like a drum with a blanket over it.

* At theta = 0 the expression genuinely does vanish for m >= 1: an on-axis mic
  really does hear only the breathing modes.  Off axis the petal modes stop
  cancelling, which is most of what a tom is.

* beta -> j is COINCIDENCE, where the sheet's phase speed reaches the speed of
  sound.  The expression is 0/0 there and its limit is pi a^2 J_m'(j)^2 —
  finite and large.  That is a real radiation peak, not a division to dodge, and
  it is why a cymbal is loud and a drumhead is not.  A stiff sheet reaches it
  inside the audio band (a steel foil at j = 144).
"""
import numpy as np
from scipy.special import jv

from .constants import C_AIR

_COINCIDENCE_TOL = 1e-3          # relative width of the l'Hopital window


def far_field_weight(a, m, j, jp, beta):
    """Rayleigh weight, with the coincidence limit handled."""
    m = np.asarray(m)
    j = np.asarray(j, dtype=float)
    jp = np.asarray(jp, dtype=float)
    beta = np.asarray(beta, dtype=float)
    d = beta * beta - j * j
    near = np.abs(d) < _COINCIDENCE_TOL * j * j
    JM = np.empty_like(beta)
    for mm in np.unique(m):
        sel = (m == mm)
        JM[sel] = jv(int(mm), beta[sel])
    safe = np.where(near, 1.0, d)                 # avoid 0/0 before selecting
    out = 2 * np.pi * a * a * j * jp * JM / safe
    return np.where(near, np.pi * a * a * jp * jp, out)


def beta_of(omega, a, theta):
    return (omega * a / C_AIR) * np.sin(theta)
