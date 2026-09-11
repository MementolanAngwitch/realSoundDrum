"""
The two precomputed tables, and their generators.

Neither is source.  Both are pure functions of the maths here, so they are
regenerable and safe to delete — but they are slow enough that recomputing them
per drum would be absurd, and the air table (~29000 numerical integrals) is slow
enough that it must be computed once, ever.

    modes_full.json   the clamped-disc mode table    (~1.1 MB, ~1 s)
    air_table.npz     the air added-mass surface     (~230 KB, ~60 s)

Run `python build_tables.py` at the repo root to make both.

------------------------------------------------------------------ mode table

For a clamped circular sheet,

    Phi_mn(r,theta) = J_m(j_mn r/a) cos(m theta),   j_mn = n-th zero of J_m

so everything downstream needs three numbers per mode: m, j_mn, J_m'(j_mn).

Store J_m'(j_mn) SIGNED.  The normalisation integral only wants its square,

    INT INT Phi^2 dA = eps_m (pi a^2/2) J_m'(j_mn)^2,   eps_m = 2 if m==0 else 1

but the radiation weight wants the bare J_m'(j), and its sign sets the relative
phase of the partials.  Recovering it as sqrt() of the square loses that.

WHICH MODES.  Two competing needs, and the asymmetry between them is physical:

  * A slow, heavy sheet (bass-drum head, c ~ 56 m/s, so r = c/c_air ~ 0.16)
    reaches 16 kHz only at very large j — but the radiation weight carries
    J_m(beta) with beta = r j sin(theta), and J_m dies exponentially once
    m >> beta.  High angular orders on a slow sheet are acoustically invisible.

  * A fast, light sheet (tight 3 mil reso head, c ~ 284 m/s, r ~ 0.83) needs
    every m, because beta is comparable to j — but it runs out of audio band by
    j ~ 65, so it never needs large j.

Hence: every m below J_ALL, and above it only m <= M_SLOPE j + M_INT.  That is
2427 modes instead of the ~7000 a naive j < J_MAX cut would give.
"""
import os
import json
import numpy as np
from scipy.special import jn_zeros, jv, jvp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

MODES_PATH = os.path.join(_ROOT, 'modes_full.json')
AIR_PATH   = os.path.join(_ROOT, 'air_table.npz')

# --- mode-table selection rule ---------------------------------------------
J_MAX, J_ALL = 172.0, 70.0
M_SLOPE, M_INT = 0.42, 7.0
FR_STEP = 0.05

# --- air-table grid ---------------------------------------------------------
R_LO, R_HI, R_N = 0.04, 0.92, 12
QUAD_N = 900


# ---------------------------------------------------------------- generation
def build_modes(path=MODES_PATH):
    rows = []
    for m in range(0, 120):
        z = jn_zeros(m, 120)
        if z[0] > J_MAX:
            break
        for n, j in enumerate(z, 1):
            if j > J_MAX:
                break
            if j < J_ALL or m <= M_SLOPE * j + M_INT:
                rows.append((m, n, float(j), float(jvp(m, j))))
    rows.sort(key=lambda r: r[2])
    M  = np.array([r[0] for r in rows])
    N  = np.array([r[1] for r in rows])
    J  = np.array([r[2] for r in rows])
    JP = np.array([r[3] for r in rows])              # SIGNED
    FR = np.round(np.arange(0, 1.0, FR_STEP), 4)
    # Phi on the strike axis; cos(m theta) is applied by the caller.
    PHI = np.array([[jv(m, j * fr) for m, j in zip(M, J)] for fr in FR])
    # encoding pinned: Python on Windows defaults text I/O to the system code
    # page, not UTF-8, and JSON is UTF-8 by definition.
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(dict(M=M.tolist(), N=N.tolist(), J=J.tolist(), JP=JP.tolist(),
                       FR=FR.tolist(), PHI=PHI.tolist()), fh)
    return M, J


def build_air(M, J, path=AIR_PATH):
    from .air import added_mass_coeff
    rs = np.linspace(R_LO, R_HI, R_N)
    A = np.array([added_mass_coeff(M, J, np.full(len(J), r), n_quad=QUAD_N)
                  for r in rs])
    np.savez(path, rs=rs, A=A)
    return A


# ------------------------------------------------------------------- loading
_MODES = None
_AIR = None


def _missing(path):
    raise FileNotFoundError(
        "%s is missing.  It is generated, not source -- run\n"
        "    python build_tables.py\n"
        "at the repository root (about a minute, mostly the air table)."
        % os.path.basename(path))


def modes():
    """The mode table as a dict of numpy arrays: M, J, JP, FR, PHI."""
    global _MODES
    if _MODES is None:
        if not os.path.exists(MODES_PATH):
            _missing(MODES_PATH)
        with open(MODES_PATH, encoding='utf-8') as fh:
            d = json.load(fh)
        _MODES = {k: np.array(v) for k, v in d.items()}
    return _MODES


def air_grid():
    """(rs, A) — the added-mass coefficient on its r grid."""
    global _AIR
    if _AIR is None:
        if not os.path.exists(AIR_PATH):
            _missing(AIR_PATH)
        d = np.load(AIR_PATH)
        # The two tables are indexed by the same mode number, so a stale one
        # beside a fresh one is not an error anywhere -- it silently hands each
        # mode a different mode's added mass.  build_tables.py checks this when
        # it writes them; this checks it when they are read, which is where a
        # half-finished rebuild actually shows up.
        n_air = d['A'].shape[1]
        n_mod = len(modes()['J'])
        if n_air != n_mod:
            raise RuntimeError(
                "table mismatch: air_table.npz has %d modes, modes_full.json has "
                "%d.  One was rebuilt without the other -- delete both and run\n"
                "    python build_tables.py" % (n_air, n_mod))
        _AIR = (d['rs'], d['A'])
    return _AIR


def air_lookup(idx, r):
    """A(mode, r) by linear interpolation on the precomputed grid.

    The quadrature behind it is ~1e4x slower and only ~0.1% more accurate, so
    the table is the right call everywhere except when building the table.
    """
    rs, A = air_grid()
    x = np.clip((r - rs[0]) / (rs[1] - rs[0]), 0, len(rs) - 1.0001)
    k = x.astype(int)
    t = x - k
    return A[k, idx] * (1 - t) + A[k + 1, idx] * t
