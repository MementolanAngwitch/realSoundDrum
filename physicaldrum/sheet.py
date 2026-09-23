"""
One clamped circular sheet: which modes it has, and what each one does.

    omega^2 = (T k^2 + D k^4) / sigma_eff,    k = j_mn/a,   D = E h^3/12(1-nu^2)

The k^4 term is BENDING STIFFNESS and it is not a small correction.  On a 10 mil
Mylar head it is already half the restoring force by j = 120, and 98% on a steel
foil:

        j      T k^2      D k^4    stiffness share
       10   1.29e+07   8.99e+04         0.7%
       80   8.26e+08   3.68e+08        30.8%
      120   1.86e+09   1.86e+09        50.1%
      170   3.73e+09   7.51e+09        66.8%

A model carrying only T k^2 can be a drum and nothing else.  The two terms are
kept in separate columns (omT2, omD2) all the way through, because tension
modulation scales only the first.

Caveat, stated once and worth remembering: the mode SHAPES here are the membrane
ones.  A stiffness-dominated sheet is approximated, not solved.  `stiffness_share`
on the result says how far into that approximation a given sheet is.

DAMPING is a constant loss factor,

    sigma_i = sigma_0 + eta omega_i/2

i.e. constant Q.  A polymer loses a fixed FRACTION of the strain energy per
cycle, not a fixed amount per second.  eta ~ 0.015 for Mylar lands inside the
measured t60 range for a real tom in every octave from 200 Hz to 3 kHz.  The
sigma_a + sigma_b f^2 form often quoted is a fitted curve with no mechanism, and
measured against a render it is ~3.5x too weak at 200 Hz and ~2x too strong at
3 kHz — low partials sustaining while the highs vanish, which is exactly what
"metallic" means.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import jv

from .constants import C_AIR, RHO_AIR, MAXM, KEEP_DB, J_KEEP
from .materials import material
from .radiation import far_field_weight, beta_of
from . import tables

MIL = 2.54e-5            # one mil, in metres


@dataclass
class Sheet:
    a: float             # radius, m
    T: float             # tension, N/m
    h: float             # thickness, m
    D: float             # bending rigidity, N m
    smu: float           # bare surface density, kg/m^2
    mat: object
    idx: np.ndarray      # index into the global mode table
    m: np.ndarray
    j: np.ndarray
    jp: np.ndarray
    k: np.ndarray
    se: np.ndarray       # air-loaded surface density, per mode
    omT2: np.ndarray     # T k^2 / se
    omD2: np.ndarray     # D k^4 / se
    om: np.ndarray
    f: np.ndarray
    norm: np.ndarray     # INT INT Phi^2 dA
    R: np.ndarray        # INT INT Phi dA -- zero unless m == 0
    sig: np.ndarray
    Drad: np.ndarray     # far-field weight at the mic angle
    mass: np.ndarray     # se * norm

    @property
    def n(self):
        return len(self.j)

    @property
    def c(self):
        """Long-wave (tension-only) speed -- what sets the contact time."""
        return math.sqrt(self.T / self.smu)

    @property
    def stiffness_share(self):
        """Bending's share of the restoring force at the top of the band."""
        return float(self.omD2[-1] / (self.omT2[-1] + self.omD2[-1]))

    def phi_at(self, i, fr, th=0.0):
        """Mode i evaluated at radius fr*a, angle th."""
        return jv(int(self.m[i]), self.j[i] * fr) * math.cos(self.m[i] * th)


def make_sheet(a, T, h_mil, s0, eta, theta, tip, fmax, mat=0, airload=True, sides=1):
    """Select the modes this sheet has, and fill in everything per mode.

    `tip` is the striker radius in metres.  It belongs here, not only in the
    strike, because it decides which modes are audible: the acoustic prune below
    ranks on the amplitude a strike of that width actually produces.  Which
    modes matter genuinely depends on how you hit it.

    `sides` is how many faces see open air, as a half-space each.  One for a
    sheet over a sealed cavity -- the cavity side is then cavity.py's job.  Two
    for a pattern that sees open air on both faces.  'open' for a sheet with
    nothing behind it, which is two for m >= 1 but ONE for m = 0: air flows
    round the rim of an unbaffled sheet, and for breathing motion that relief is
    total -- Lamb's rigid disc carries 8 rho a^3/3 moving broadside in free air,
    exactly the one-face baffled value.  Patterns with nodal diameters cancel
    their own flow before it reaches the rim, and see both faces.
    """
    M = material(mat)
    h = h_mil * MIL
    smu = M.rho * h
    D = M.E * h ** 3 / (12 * (1 - M.nu ** 2))

    tb = tables.modes()
    jj, mm, jp = tb['J'], tb['M'], tb['JP']
    kk = jj / a
    w2 = (T * kk * kk + D * kk ** 4) / smu
    om0 = np.sqrt(w2)

    # --- acoustic prune ----------------------------------------------------
    # Reaching 16 kHz on a tom naively needs ~2300 modes.  But D ~ J_m(beta)
    # with beta = r j sin(theta), and J_m dies exponentially once m >> beta, so
    # high angular orders are acoustically invisible: only ~290 of those 2300
    # clear -50 dB.  Keep those, plus everything below J_KEEP -- quiet, but they
    # carry the tension sum S.
    eps = np.where(mm == 0, 2.0, 1.0)
    nrm = eps * (np.pi * a * a / 2) * jp * jp
    Dr = np.abs(far_field_weight(a, mm, jj, jp, beta_of(om0, a, theta)))
    amp = Dr * w2 * np.exp(-0.5 * (kk * tip) ** 2) / (smu * nrm)

    in_band = (om0 / (2 * np.pi) < fmax) | (jj <= 8)
    amp = np.where(in_band, amp, 0.0)
    keep = in_band & ((amp > amp.max() * 10 ** (-KEEP_DB / 20)) | (jj < J_KEEP))
    idx = np.flatnonzero(keep)[:MAXM]

    m_, j_, jp_ = mm[idx], jj[idx], jp[idx]
    k_ = j_ / a

    # --- air loading, self-consistent --------------------------------------
    from .air import loaded_density
    if airload:
        sd = np.where(m_ == 0, 1.0, 2.0) if sides == 'open' else float(sides)
        se, omT2, omD2 = loaded_density(k_, idx, T, D, smu, a, sides=sd)
    else:
        se = np.full_like(k_, smu, dtype=float)
        omT2, omD2 = T * k_ ** 2 / se, D * k_ ** 4 / se

    om = np.sqrt(omT2 + omD2)
    eps_ = np.where(m_ == 0, 2.0, 1.0)
    norm = eps_ * (np.pi * a * a / 2) * jp_ * jp_
    R = np.where(m_ == 0, 2 * np.pi * a * a * (-jp_) / j_, 0.0)
    sig = s0 + (eta * 1e-3) * om / 2                  # hysteretic: constant Q
    Drad = far_field_weight(a, m_, j_, jp_, beta_of(om, a, theta))

    return Sheet(a=a, T=T, h=h, D=D, smu=smu, mat=M, idx=idx, m=m_, j=j_, jp=jp_,
                 k=k_, se=se, omT2=omT2, omD2=omD2, om=om, f=om / (2 * np.pi),
                 norm=norm, R=R, sig=sig, Drad=Drad, mass=se * norm)
