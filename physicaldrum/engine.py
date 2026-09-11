"""
Assemble a struck sheet, and render one strike.

`build` turns a parameter set into a flat bank of oscillators; `render` runs it.
This mirrors the browser engine in `web/physicalDrumBeta.html` function for
function, deliberately, so that `tests/test_parity.py` can assert the two agree.
If you change the physics here, change it there, and run the parity test.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .constants import (SR, DUR, RHO_AIR, K_AIR, RAD_C, MIC_RESO, MIC_PORT,
                        PORT_LEN, PICKUP_FR, PICKUP_TH, DEAD_DB, OM_MOD,
                        GAMMA_MAX, RMS_TARGET, RMS_WIN, DRIVE_MAX)
from .sheet import make_sheet
from .cavity import secular, split_ties, eigenvector
from . import tables


# --------------------------------------------------------------- parameters
@dataclass
class Params:
    """Everything about one struck sheet.  Lengths in mm unless noted."""
    mat: object = 0
    a: float = 0.164          # radius, METRES
    T: float = 3471           # tension, N/m
    h: float = 10.0           # thickness, mil
    s0: float = 1.2           # damping floor, 1/s
    eta: float = 16           # loss factor x1e-3
    P: float = 5.83           # strike impulse, mN.s
    w: float = 3.5            # striker tip radius, mm
    hard: float = 0.0         # striker softness: a contact-time floor, ms
    s: int = 8                # strike position, index into the radial grid
    th: float = 0.0           # strike angle, rad
    mic: int = 1              # 1 = microphone, 0 = contact pickup
    d: float = 210            # cavity depth, mm  (>= 3000 means open back)
    Tr: float = 0.85          # second sheet tension ratio; 0 = no second sheet
    hr: float = 7.5           # its thickness, mil
    mat2: object = None       # its material; None = same as the first
    port: float = 0.0         # vent radius, mm
    ang: float = 60           # mic angle off axis, degrees
    air: int = 1              # air loading on/off
    fmax: float = 16000       # band ceiling, Hz
    name: str = ''


@dataclass
class Drum:
    """A flat oscillator bank, ordered by lifetime, plus what it came from."""
    om: np.ndarray
    sig: np.ndarray
    out: np.ndarray
    v: np.ndarray
    kern: np.ndarray
    mi: np.ndarray
    life: np.ndarray
    omT2: np.ndarray
    omD2: np.ndarray
    beta: float
    tauL: float
    tau: float
    iF1: int
    f1: float
    fC: np.ndarray
    nU: int
    nC: int
    info: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.om)


def _sinc(x):
    return np.where(x == 0, 1.0, np.sin(np.pi * x) / np.where(x == 0, 1.0, np.pi * x))


def _contact_filter(om, tau):
    """Spectrum of a half-sine force pulse of duration tau.

    A symmetric pulse has a real transform up to a pure delay, so applying this
    as a magnitude correction to an instantaneous velocity kick gives the same
    waveform as the real time-domain pulse, shifted.  (Checked; it is not the
    reason the transient is peaky.)
    """
    if tau <= 0:
        return np.ones_like(om)
    x = om * tau / np.pi
    return np.clip(np.abs(_sinc(om * tau / (2 * np.pi)) / (1 - x * x + 1e-9)), 0, 1)


# -------------------------------------------------------------------- build
def build(p):
    theta = p.ang * np.pi / 180
    tip = p.w * 1e-3
    tb = tables.modes()
    FR = tb['FR']

    B = make_sheet(p.a, p.T, p.h, p.s0, p.eta, theta, tip, p.fmax, p.mat, p.air != 0)
    two = p.Tr > 0 and 0 < p.d < 3000
    R = (make_sheet(p.a, p.T * p.Tr, p.hr, p.s0 * 1.3, p.eta * 1.25, theta, tip,
                    p.fmax, p.mat if p.mat2 is None else p.mat2, p.air != 0)
         if two else None)

    # The tension sum S = INT |grad u|^2 dA only converges once the mode set can
    # RESOLVE the contact patch: a point force on a membrane has a 1/r slope, so
    # S diverges logarithmically until modes out to j ~ 3a/w are present.  Below
    # that the pitch bend is a truncation artifact, not physics -- measured on a
    # 16" tom, a 1.5 mm tip gave gamma = 1.166 at 200 modes and 2.000 at 1500.
    j_top = float(B.j.max())
    w_min = 3 * p.a / max(j_top, 1.0)
    tip_limited = tip < w_min
    if tip_limited:
        tip = w_min

    # Contact time: the dent spreads at the sheet's wave speed, so the tip stays
    # down about as long as the sheet needs to get out from under it -- unless
    # the striker itself is soft, in which case its own compression is slower.
    tau = max(2 * tip / B.c, (p.hard or 0.0) * 1e-3)

    # ---- uncoupled block: every m >= 1 mode of the struck sheet -------------
    U = np.flatnonzero(B.m >= 1)
    roll = np.exp(-0.5 * (B.k[U] * tip) ** 2)
    cf = _contact_filter(B.om[U], tau)
    mi = np.array([B.phi_at(i, FR[p.s], p.th) for i in U])
    omU = B.om[U]
    sigU = B.sig[U] + RAD_C * omU ** 2 * B.Drad[U] ** 2 / (B.se[U] * B.norm[U])
    if p.mic:
        outU = -B.Drad[U] * omU ** 2
    else:
        outU = np.array([B.phi_at(i, FR[PICKUP_FR], PICKUP_TH) for i in U])
    vU = mi * roll * cf / (B.se[U] * B.norm[U])
    kern = B.k[U] ** 2 * B.norm[U]

    # ---- coupled block: m == 0 of both sheets, plus the vent ----------------
    om2, mass, aV, oR, sg, isB, hidx = [], [], [], [], [], [], []
    for i in np.flatnonzero(B.m == 0):
        om2.append(B.om[i] ** 2); mass.append(B.mass[i]); aV.append(B.R[i])
        oR.append(B.Drad[i] if p.mic else B.phi_at(i, FR[PICKUP_FR], 0.0))
        sg.append(B.sig[i]); isB.append(1); hidx.append(int(i))
    if R is not None:
        for i in np.flatnonzero(R.m == 0):
            om2.append(R.om[i] ** 2); mass.append(R.mass[i]); aV.append(R.R[i])
            oR.append(MIC_RESO * R.Drad[i] if p.mic else 0.0)
            sg.append(R.sig[i]); isB.append(0); hidx.append(-1)

    depth = max(p.d, 1) * 1e-3
    V = np.pi * p.a * p.a * depth
    K = 0.0 if p.d >= 3000 else K_AIR / V
    pr = (p.port or 0.0) * 1e-3
    if pr > 0 and K > 0:
        Ap = np.pi * pr * pr
        om2.append(0.0); mass.append(RHO_AIR * Ap * PORT_LEN * pr); aV.append(-Ap)
        oR.append(MIC_PORT * Ap if p.mic else 0.0)
        sg.append(8.0); isB.append(0); hidx.append(-1)

    om2 = np.array(om2); mass = np.array(mass); aV = np.array(aV)
    oR = np.array(oR); sg = np.array(sg)
    isB = np.array(isB); hidx = np.array(hidx)
    order = np.argsort(om2)
    om2, mass, aV, oR, sg, isB, hidx = (x[order] for x in
                                        (om2, mass, aV, oR, sg, isB, hidx))
    om2 = split_ties(om2)
    nC = len(om2)
    w = aV / np.sqrt(mass)
    lam = secular(om2, w * w, K) if K > 0 else om2.copy()

    omC = np.sqrt(np.maximum(lam, 0))
    fC = omC / (2 * np.pi)
    outC = np.zeros(nC); vC = np.zeros(nC); sigC = np.zeros(nC)
    partB = np.zeros(nC); tShare = np.zeros(nC)
    bT = np.where(isB == 1, 0.0, 0.0)
    for c in range(nC):
        z = eigenvector(om2, w, lam[c]) if K > 0 else np.eye(nC)[c]
        xp = z / np.sqrt(mass)
        outC[c] = float(oR @ xp)
        vol = float(aV @ xp)
        sigC[c] = float((z * z) @ sg) + RAD_C * omC[c] ** 2 * vol ** 2 / B.smu
        pb, F, ft, fw = 0.0, 0.0, 0.0, 0.0
        for t in range(nC):
            q = z[t] * z[t]
            if isB[t]:
                i = hidx[t]
                pb += q
                F += xp[t] * B.phi_at(i, FR[p.s], 0.0) \
                     * math.exp(-0.5 * (B.k[i] * tip) ** 2) \
                     * float(_contact_filter(np.array([B.om[i]]), tau)[0])
                # split the coupled eigenvalue into the part tension can move and
                # the part it cannot, in proportion to the sheet modes behind it
                ft += q * B.omT2[i] / (B.omT2[i] + B.omD2[i])
                fw += q
            else:
                fw += q
        partB[c] = pb
        vC[c] = F
        tShare[c] = ft / fw if fw > 0 else 1.0
    outC = -outC * omC ** 2

    # ---- merge, then order by LIFETIME -------------------------------------
    # With constant Q a 10 kHz mode is inaudible after ~25 ms while the
    # fundamental runs a second, so most modes are dead for most of the render.
    # Sorting by death sample lets the loop shrink its active set rather than
    # grinding zeros -- about 15x, and exact rather than an approximation.
    om = np.concatenate([omU, omC])
    sig = np.concatenate([sigU, sigC])
    out = np.concatenate([outU, outC])
    v = np.concatenate([vU, vC])
    kn = np.concatenate([kern, np.zeros(nC)])
    mif = np.concatenate([mi, np.zeros(nC)])
    oT = np.concatenate([B.omT2[U], lam * tShare])
    oD = np.concatenate([B.omD2[U], lam * (1 - tShare)])

    life = np.minimum(DEAD_DB / 20 * math.log(10) / np.maximum(sig, 1e-6), DUR) * SR
    o = np.argsort(-life, kind='stable')
    om, sig, out, v, kn, mif, life, oT, oD = (x[o] for x in
                                              (om, sig, out, v, kn, mif, life, oT, oD))

    beta = B.mat.E * B.h / (2 * np.pi * p.a * p.a * p.T)
    tauL = 2 * p.a / math.sqrt(B.mat.E / B.mat.rho)

    cand = fC[(fC > 20) & (partB > 0.30)]
    f1 = float(cand.min()) if len(cand) else 1e9
    if len(omU) and omU[0] / (2 * np.pi) < f1:
        f1 = float(omU[0] / (2 * np.pi))
    iF1 = int(np.argmin(np.abs(om / (2 * np.pi) - f1)))

    mx = float(np.abs(out * v).max()) if len(out) else 0.0
    n_rad = int((np.abs(out * v) > mx * 1e-2).sum())
    f_vent = 0.0
    if pr > 0 and K > 0:
        q = fC[(partB < 0.15) & (fC > 5)]
        f_vent = float(q.min()) if len(q) else 0.0
    f_air = (0.0, 0.0)
    if R is not None:
        q = np.sort(fC[(partB > 0.15) & (partB < 0.99) & (fC > 20)])
        if len(q) > 1:
            f_air = (float(q[0]), float(q[1]))

    info = dict(material=B.mat.name, n_modes=B.n, f_top=float(B.f.max()),
                stiffness=B.stiffness_share, air_fraction=float(B.se[0] / B.smu - 1),
                tip_eff=tip, tip_limited=bool(tip_limited), two_sheet=R is not None,
                vented=bool(pr > 0 and K > 0), f_vent=f_vent, f_air=f_air,
                volume=V, c=B.c, sheet=B, reso=R)

    return Drum(om=om, sig=sig, out=out, v=v, kern=kn, mi=mif, life=life,
                omT2=oT, omD2=oD, beta=beta, tauL=tauL, tau=tau, iF1=iF1,
                f1=f1 if np.isfinite(f1) else 0.0, fC=fC,
                nU=len(omU), nC=nC, info=info)


# ------------------------------------------------------------------- render
def render(d, impulse=None, vel=1.0, dur=DUR, sr=SR, limiter=True):
    """One strike.  `impulse` in N.s; defaults to the drum's own P if given.

    State is (q, v) -- position and velocity -- advanced by the exact
    damped-oscillator rotation.  The two-pole form q[n+1] = a1 q[n] - a2 q[n-1]
    is exact at a FIXED frequency, but tension modulation changes the frequency
    every few samples, and swapping a1 while keeping both past samples is not
    energy-consistent: the pair encodes an amplitude that means something
    different under the new frequency.  Position and velocity are physical, so
    the same rotation with a new omega is simply the same state seen by a
    different oscillator.
    """
    n = d.n
    N = int(dur * sr)
    dt = 1.0 / sr
    y = np.zeros(N)
    if n == 0:
        return y, dict(gmax=1.0, bend=0.0, dent=0.0, drive=1.0, clipped=False)

    P = (impulse if impulse is not None else 0.0) * vel
    dec = np.exp(-d.sig * dt)
    w2 = d.om ** 2
    wd = np.sqrt(np.maximum(w2 - d.sig ** 2, 1e-9))
    cc = dec * np.cos(wd * dt)
    sw = dec * np.sin(wd * dt) / wd
    q = np.zeros(n)
    v = P * d.v.copy()                       # the strike is a velocity kick

    # The tension model says the whole sheet feels one raised tension.  It
    # cannot: a rise in tension is carried across the sheet by the IN-PLANE wave
    # at c_L = sqrt(E/rho), so global tension lags local stretch by about one
    # crossing time.  Treating it as instantaneous lets the first few samples --
    # when every mode is still in phase and the modal sum for S is at its most
    # coherent and least converged -- set gamma for the whole note.
    a_lag = 1 - math.exp(-dt / max(d.tauL, dt))
    modulatable = d.om <= OM_MOD

    g, gmax, dent, act, clipped, Ss = 1.0, 1.0, 0.0, n, False, 0.0
    n_fast = int(0.001 * sr)
    for t in range(1, N):
        while act > 0 and d.life[act - 1] <= t:
            act -= 1
        sl = slice(0, act)
        # qi must be a COPY: q[sl] is a view, so writing q[sl] first would feed
        # the already-updated position into the velocity line.  In the JS these
        # are scalar locals and the hazard does not exist.
        qi = q[sl].copy()
        vi = v[sl]
        sg = d.sig[sl]
        q[sl] = cc[sl] * qi + sw[sl] * (vi + sg * qi)
        v[sl] = cc[sl] * vi - sw[sl] * (w2[sl] * qi + sg * vi)
        qn = q[sl]
        y[t] = float(d.out[sl] @ qn)
        u = float(d.mi[sl] @ qn)
        dent = max(dent, abs(u))
        S = float(d.kern[sl] @ (qn * qn))
        Ss += (S - Ss) * a_lag
        gi = math.sqrt(1 + d.beta * Ss)
        if not gi < GAMMA_MAX:
            gi, clipped = GAMMA_MAX, True
        gmax = max(gmax, gi)
        L = 4 if t < n_fast else 8
        if t % L == 0 and abs(gi - g) > 1e-6:
            g = gi
            sel = np.zeros(n, bool)
            sel[:act] = modulatable[:act]
            # gamma stretches the sheet; it does not change its bending
            # rigidity, so it multiplies only the k^2 term.
            w2[sel] = g * g * d.omT2[sel] + d.omD2[sel]
            wd[sel] = np.sqrt(np.maximum(w2[sel] - d.sig[sel] ** 2, 1e-9))
            cc[sel] = dec[sel] * np.cos(wd[sel] * dt)
            sw[sel] = dec[sel] * np.sin(wd[sel] * dt) / wd[sel]

    pk = float(np.abs(y).max())
    if pk > 0:
        y *= 0.92 / pk
    drive = 1.0
    if limiter:
        W = min(int(RMS_WIN * sr), N)
        rms = float(np.sqrt((y[:W] ** 2).mean()))
        drive = RMS_TARGET / rms if rms > 1e-9 else 1.0
        drive = min(max(drive, 1.0), DRIVE_MAX)
        if drive > 1.0001:
            y = np.tanh(y * drive) / math.tanh(1.0) * 0.92
        pk2 = float(np.abs(y).max())
        if pk2 > 0.97:
            y *= 0.97 / pk2
    fd = min(600, N)
    y[N - fd:] *= np.arange(fd)[::-1] / fd

    # gamma is not the pitch bend.  It multiplies only the tension term, so on a
    # stiffness-dominated sheet a large gamma moves the partials hardly at all.
    bT, bD = d.omT2[d.iF1], d.omD2[d.iF1]
    bend = (1200 * math.log2(math.sqrt((gmax * gmax * bT + bD) / (bT + bD)))
            if bT + bD > 0 else 0.0)
    return y, dict(gmax=gmax, bend=bend, dent=dent, drive=drive, clipped=clipped)
