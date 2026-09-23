"""
The air sealed between (or behind) the sheets, with its real mode set.

WHY NOT ONE LUMPED COMPLIANCE.  Treating the enclosed air as a single uniform
pressure couples a sheet mode through INT INT Phi dA, which is zero for every
m >= 1.  In that model the struck sheet cannot drive the far sheet at any
angular order above zero, so the far sheet only ever breathes -- and snare wires
resting on it have nothing but one slow sine to rattle against.  The lumped
model is right below the first non-uniform cavity mode, 1.8412 c/(2 pi a) =
565 Hz on a 14" drum, and wrong above it.

THE MODES.  A rigid cylinder, radius a, depth L, sheets at x = 0 and x = L:

    Psi_mqs = J_m(chi_mq r/a) cos(m theta) cos(s pi x/L),   chi_mq: zeros of J_m'
    omega_mqs = c sqrt((chi/a)^2 + (s pi/L)^2)
    Lambda    = INT Psi^2 dV = ang . rad . (L if s == 0 else L/2)
    ang = 2 pi (m = 0) or pi,   rad = (a^2/2) J_m(chi)^2 (1 - m^2/chi^2)   (a^2/2 at chi = 0)

A sheet mode couples only to cavity modes of the SAME m, so everything is block
diagonal in angular order.  The overlap is Lommel again -- the same closed form
as the far field and the added mass, for the same reason (J_m(j) = 0 at the
clamped rim, J_m'(chi) = 0 at the rigid wall):

    G = INT Phi Psi dA = ang a^2 (-j J_m'(j) J_m(chi)) / (j^2 - chi^2)

and chi = 0, m = 0 gives G = INT Phi dA, so the old rank-one model is the
q = s = 0 term of this one.  That is the fourth use of one integral.

THE SYMMETRIC FORM.  Projecting the wave equation with the moving-wall boundary
condition gives, both sheets taken positive INTO the cavity,

    M q'' + K q = -G P                          (sheets)
    (Lambda/c^2)(P'' + omega_n^2 P) = rho G^T q''     (air)

which is not symmetric -- the coupling sits in the stiffness one way and the mass
the other, and an eigensolver hands back vectors that are not mass-orthogonal.
Substitute  P_n = kappa_n (g_n^T q - eta_n),  kappa_n = rho c^2/Lambda_n :

    M q'' + [K + SUM kappa g g^T] q - SUM kappa g eta = 0
    mu_n eta_n'' + kappa_n eta_n - kappa_n g_n^T q     = 0,    mu_n = kappa_n/omega_n^2

Mass diagonal, stiffness symmetric, strain energy SUM kappa (g^T q - eta)^2 >= 0.
eta_n is the volume of air the mode has actually moved; g^T q - eta is what is
left over to compress.  For the uniform mode omega = 0, mu is infinite, eta stays
at zero, and what remains is kappa_0 R R^T -- the rank-one model, exactly.

TWO KINDS OF CAVITY MODE.  Below F_DYN a cavity mode is kept as a real degree of
freedom: it can resonate with the sheets.  Above it the air cannot keep up with
anything the sheets do, eta follows g^T q quasi-statically, and the mode leaves
behind only an ADDED MASS mu g g^T.  Summed over every axial order s that is a
closed form:

    SUM_s mu_s g g^T = rho G G^T / (ang rad alpha) . { coth(alpha L)  same sheet
                                                      { csch(alpha L)  opposite,    alpha = chi/a

The csch is the evanescent field reaching across the drum: a pattern of radial
wavenumber alpha is felt at the far sheet with weight e^{-alpha L}, which is why
only the broad, low-order patterns of the struck sheet get through.  The coth
-> 1 limit is a half-space on the cavity side, and its value there,
rho . INT Phi^2 dA / k, is the infinite-plane air layer -- the cross-check that
found the factor of two in air.py.
"""
import math

import numpy as np
from scipy.special import jv, jnp_zeros
from scipy.linalg import eigh

from .constants import RHO_AIR, C_AIR

_TIE = 1e-7          # relative split for repeated roots (secular reference)
_BISECT = 110


# ------------------------------------------------------------ cavity modes
def neumann_zeros(m, chi_max):
    """Zeros of J_m' up to chi_max -- the rigid-wall radial wavenumbers.

    For m = 0 this includes chi = 0, the uniform pattern.  (J_0' = -J_1, so the
    rest are the zeros of J_1.)
    """
    n = int(chi_max / math.pi) + 4
    z = np.asarray(jnp_zeros(int(m), n), dtype=float)
    z = z[z <= chi_max]
    return np.concatenate([[0.0], z]) if m == 0 else z


def ang_of(m):
    return 2 * math.pi if m == 0 else math.pi


def rad_norm(m, chi, a):
    """INT_0^a J_m(chi r/a)^2 r dr, for chi a zero of J_m'."""
    if chi == 0.0:
        return a * a / 2
    return (a * a / 2) * jv(m, chi) ** 2 * (1 - m * m / (chi * chi))


def overlap(m, j, jp, chi, a):
    """G = INT Phi_mn Psi_mq dA over one face (axial factor not included)."""
    return ang_of(m) * a * a * (-j * jp * jv(m, chi)) / (j * j - chi * chi)


# ------------------------------------------------------------ one block
def block(m, heads, a, L, f_dyn, chi_pad, vent=None, q_cav=50.0, s0_cav=1.0,
          static=True):
    """Assemble the coupled problem for one angular order.

    heads : list of dicts with keys j, jp, face (+1 near sheet, -1 far sheet),
            M (one-sided loaded modal mass), K (modal stiffness), R (INT Phi dA)
    vent  : (mass, area) or None -- a slug of air in the shell wall, which sees
            only the uniform pressure
    static: fold the cavity modes above f_dyn in as added mass.  With
            static=False and f_dyn=0 only the uniform compliance is left, which
            is the old rank-one model exactly.
    Returns (K, M, kinds, sig_extra, dyn) where kinds[t] is 'h', 'c' or 'v' and
    sig_extra carries the cavity DOFs' own damping.
    """
    nh = len(heads)
    j = np.array([h['j'] for h in heads], dtype=float)
    jp = np.array([h['jp'] for h in heads], dtype=float)
    face = np.array([h['face'] for h in heads], dtype=float)
    Mh = np.array([h['M'] for h in heads], dtype=float)
    Kh = np.array([h['K'] for h in heads], dtype=float)

    ang = ang_of(m)
    j_top = float(j.max()) if nh else 0.0
    chis = neumann_zeros(m, j_top + chi_pad) if (static or f_dyn > 0) else []
    om_dyn = 2 * math.pi * f_dyn
    V = math.pi * a * a * L

    # dynamic cavity modes, and the static added mass of everything else
    dyn = []                                    # (chi, s, omega, Lambda, g)
    Madd = np.zeros((nh, nh))
    same = np.equal.outer(face, face)
    for chi in chis:
        rn = rad_norm(m, chi, a)
        if rn <= 0:
            continue
        G0 = np.array([overlap(m, jj, pp, chi, a) for jj, pp in zip(j, jp)])
        GG = np.outer(G0, G0)
        if chi == 0.0:
            # s >= 1 of the uniform radial pattern: a 1-D air column, plug flow
            tot = RHO_AIR * GG / (ang * rn) * np.where(same, L / 3.0, -L / 6.0)
        else:
            al = chi / a
            x = al * L
            if x > 40:                           # csch underflows; coth -> 1
                tot = RHO_AIR * GG / (ang * rn * al) * np.where(same, 1.0, 0.0)
            else:
                tot = RHO_AIR * GG / (ang * rn * al) * np.where(
                    same, 1.0 / math.tanh(x), 1.0 / math.sinh(x))
        # peel off the axial orders that are kept as real degrees of freedom
        s = 0 if chi > 0 else 1
        while True:
            om = C_AIR * math.hypot(chi / a, s * math.pi / L)
            if om >= om_dyn:
                break
            lam = ang * rn * (L if s == 0 else L / 2)
            sign = np.where(face > 0, 1.0, (-1.0) ** s)
            g = G0 * sign
            kap = RHO_AIR * C_AIR ** 2 / lam
            mu = kap / (om * om)
            tot = tot - mu * np.outer(g, g)
            dyn.append((chi, s, om, lam, g))
            s += 1
        if static:
            Madd += tot

    nv = 1 if (vent is not None and m == 0) else 0
    nc = len(dyn)
    n = nh + nc + nv
    K = np.zeros((n, n))
    M = np.zeros((n, n))
    K[:nh, :nh] = np.diag(Kh)
    M[:nh, :nh] = np.diag(Mh) + Madd
    kinds = ['h'] * nh + ['c'] * nc + ['v'] * nv
    sig_extra = np.zeros(n)

    for c, (chi, s, om, lam, g) in enumerate(dyn):
        t = nh + c
        kap = RHO_AIR * C_AIR ** 2 / lam
        K[:nh, :nh] += kap * np.outer(g, g)
        K[:nh, t] = -kap * g
        K[t, :nh] = -kap * g
        K[t, t] = kap
        M[t, t] = kap / (om * om)
        sig_extra[t] = s0_cav + om / (2 * q_cav)

    if m == 0:
        # the uniform mode: pure compliance, rank one
        av = np.zeros(n)
        av[:nh] = [h['R'] for h in heads]
        if nv:
            mv, Ap = vent
            av[n - 1] = -Ap
            M[n - 1, n - 1] = mv
            sig_extra[n - 1] = 8.0
        K += (RHO_AIR * C_AIR ** 2 / V) * np.outer(av, av)

    return K, M, kinds, sig_extra, dyn


def gen_eigh(K, M):
    """K x = lam M x with X^T M X = I, eigenvalues ascending."""
    lam, X = eigh(K, M)
    return lam, X


# ------------------------------------------------------------ reference
def split_ties(om2):
    """Nudge exact duplicates apart so every bracket has width."""
    out = np.array(om2, dtype=float)
    for t in range(1, len(out)):
        lo = out[t - 1] * (1 + _TIE) + 1e-6
        if out[t] < lo:
            out[t] = lo
    return out


def secular(om2, w2, K):
    """Eigenvalues of diag(om2) + K w w^T, by bisection on the secular equation.

    No longer used by the engine.  Kept because it is the EXACT answer for the
    uniform-pressure limit, and the modal cavity must reduce to it -- see
    tests/test_physics.py.  Its roots interlace the om2, one per gap, which is
    what makes plain bisection unconditionally safe.
    """
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
