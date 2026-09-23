"""
Assemble a struck sheet, and render one strike.

`build` turns a parameter set into a flat bank of oscillators; `render` runs it.
This mirrors the browser engine in `web/physicalDrumBeta.html` function for
function, deliberately, so that `tests/test_parity.py` can assert the two agree.
If you change the physics here, change it there, and run the parity test.

THE OSCILLATOR BANK has three kinds of member.

  U  modes of the struck sheet that nothing else can reach: m >= 1 patterns too
     fine to cross the cavity (or every m >= 1 mode, with no cavity).  One
     sheet mode each, in its own modal coordinate.
  C  eigenmodes of a coupled block: for one angular order m, the sheet modes of
     both sheets that the cavity can couple, plus the cavity's own low modes,
     plus the vent (m = 0 only).  Unit modal mass.
  W  modes of the far sheet that only the wires can reach.  Unit modal mass.
     Present only when there are wires.

Everything the render needs about a member -- its frequency, damping, what the
microphone hears of it, how a strike excites it, how it shapes the tension sum,
and where the wires touch it -- is folded into per-member numbers here.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .constants import (SR, DUR, RHO_AIR, RAD_C, MIC_RESO, MIC_PORT,
                        PORT_LEN, PICKUP_FR, PICKUP_TH, DEAD_DB, OM_MOD,
                        GAMMA_MAX, RMS_TARGET, RMS_WIN, DRIVE_MAX,
                        F_DYN, F_DYN_RATIO, J_CAV_MIN, J_CAV_MAX, CHI_PAD, Q_CAV, S0_CAV,
                        WIRE_SIG, F_WIRE, WIRE_QUIET)
from .sheet import make_sheet
from .cavity import block, gen_eigh
from . import wires as W
from . import tables
from scipy.special import jv


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
    wN: int = 0               # snare strands on the second sheet; 0 = none
    wT: float = 30.0          # strand tension, N
    wMu: float = 1.6          # strand linear density, g/m
    wBed: float = 0.5         # snare bed: how far past the sheet's plane the strand anchors sit, mm
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
    nW: int = 0
    widx: np.ndarray = None        # which members the wires touch
    wshape: np.ndarray = None      # (points, len(widx)): member shape at each point
    wire: dict = None              # strand state and constants
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
    waveform as the real time-domain pulse, shifted.
    """
    om = np.asarray(om, dtype=float)
    if tau <= 0:
        return np.ones_like(om)
    x = om * tau / np.pi
    return np.clip(np.abs(_sinc(om * tau / (2 * np.pi)) / (1 - x * x + 1e-9)), 0, 1)


def _phi(S, idx, fr, th):
    """Mode shapes of sheet S (modes idx) at radius fractions fr, angles th.

    fr, th broadcast together; result has shape fr.shape + (len(idx),).
    """
    fr = np.asarray(fr, dtype=float)[..., None]
    th = np.asarray(th, dtype=float)[..., None]
    m = S.m[idx]
    out = np.empty(np.broadcast(fr, th).shape[:-1] + (len(idx),))
    for mm in np.unique(m):
        sel = np.flatnonzero(m == mm)
        out[..., sel] = jv(int(mm), S.j[idx][sel] * fr) * np.cos(mm * th)
    return out


def cavity_reach(a, L):
    """Highest j the cavity couples in full.  See constants.J_CAV_MIN."""
    return float(np.clip(max(J_CAV_MIN, 4 * a / L), J_CAV_MIN, J_CAV_MAX))


# -------------------------------------------------------------------- build
def build(p):
    theta = p.ang * np.pi / 180
    tip = p.w * 1e-3
    FR = tables.modes()['FR']
    air = p.air != 0
    mat2 = p.mat if p.mat2 is None else p.mat2

    sealed = 0 < p.d < 3000
    two = sealed and p.Tr > 0
    L = max(p.d, 1) * 1e-3

    # Which faces see open air.  A sheet over a sealed cavity has one; its
    # cavity face is cavity.py's job, except for patterns too fine to cross the
    # cavity, which see it as a half-space (B2).  With nothing behind it, see
    # sheet.make_sheet's 'open'.
    B = make_sheet(p.a, p.T, p.h, p.s0, p.eta, theta, tip, p.fmax, p.mat, air,
                   sides=1 if sealed else 'open')
    B2 = (make_sheet(p.a, p.T, p.h, p.s0, p.eta, theta, tip, p.fmax, p.mat, air,
                     sides=2) if sealed else B)
    R = R2 = None
    if two:
        args = (p.a, p.T * p.Tr, p.hr, p.s0 * 1.3, p.eta * 1.25, theta, tip,
                p.fmax, mat2, air)
        R = make_sheet(*args, sides=1)
        R2 = make_sheet(*args, sides=2)

    # The tension sum S = INT |grad u|^2 dA only converges once the mode set can
    # RESOLVE the contact patch: a point force on a membrane has a 1/r slope, so
    # S diverges logarithmically until modes out to j ~ 3a/w are present.
    j_top = float(B.j.max())
    w_min = 3 * p.a / max(j_top, 1.0)
    tip_limited = tip < w_min
    if tip_limited:
        tip = w_min

    # Contact time: the dent spreads at the sheet's wave speed, so the tip stays
    # down about as long as the sheet needs to get out from under it -- unless
    # the striker itself is soft, in which case its own compression is slower.
    tau = max(2 * tip / B.c, (p.hard or 0.0) * 1e-3)

    # ---- which sheet modes the cavity couples -------------------------------
    jcav = cavity_reach(p.a, L) if (sealed and air) else 0.0

    def coupled(S):
        return (S.m == 0) | ((S.j <= jcav) & sealed)

    inB = coupled(B)
    inR = coupled(R) if two else None
    # cavity modes live up to F_DYN_RATIO x the fastest coupled sheet pattern
    f_cpl = float(B.f[inB & (B.m > 0)].max()) if (inB & (B.m > 0)).any() else 0.0
    if two and (inR & (R.m > 0)).any():
        f_cpl = max(f_cpl, float(R.f[inR & (R.m > 0)].max()))
    f_dyn = min(F_DYN, F_DYN_RATIO * f_cpl) if f_cpl > 0 else F_DYN

    pr = (p.port or 0.0) * 1e-3
    vented = pr > 0 and sealed
    vent = (RHO_AIR * math.pi * pr * pr * PORT_LEN * pr, math.pi * pr * pr) if vented else None

    wires_on = two and p.wN > 0
    if wires_on:
        wr, wth, wls, whalf, wu = W.geometry(p.a, p.wN)

    # ---- coupled blocks ------------------------------------------------------
    orders = sorted(set(B.m[inB].tolist()) | (set(R.m[inR].tolist()) if two else set()))
    C = dict(om=[], sig=[], out=[], v=[], kern=[], mi=[], oT=[], oD=[], partB=[],
             m=[], vfrac=[], shape=[])
    for mm in orders:
        heads = []
        iB = np.flatnonzero(inB & (B.m == mm))
        for i in iB:
            heads.append(dict(j=B.j[i], jp=B.jp[i], face=1, M=B.mass[i],
                              K=B.mass[i] * B.om[i] ** 2, R=B.R[i]))
        iR = np.flatnonzero(inR & (R.m == mm)) if two else np.array([], int)
        for i in iR:
            # 1e-7 on the far sheet's stiffness: identical sheets otherwise give
            # exactly repeated eigenvalues, whose eigenvectors are any rotation
            # of each other -- and the two engines would pick different ones.
            heads.append(dict(j=R.j[i], jp=R.jp[i], face=-1, M=R.mass[i],
                              K=R.mass[i] * R.om[i] ** 2 * (1 + 1e-7), R=R.R[i]))
        nb, nr = len(iB), len(iR)
        if sealed:
            K, M, kinds, sig_x, _ = block(
                mm, heads, p.a, L, f_dyn if air else 0.0, CHI_PAD,
                vent=vent, q_cav=Q_CAV, s0_cav=S0_CAV, static=air)
        else:                                   # open back: m = 0, nothing to couple
            K = np.diag([h['K'] for h in heads])
            M = np.diag([h['M'] for h in heads])
            kinds = ['h'] * len(heads)
            sig_x = np.zeros(len(heads))
        lam, X = gen_eigh(K, M)
        n = len(kinds)
        isB = np.zeros(n); isB[:nb] = 1
        isR = np.zeros(n); isR[nb:nb + nr] = 1
        isV = np.array([k == 'v' for k in kinds], float)

        sgd = sig_x.copy()
        sgd[:nb] = B.sig[iB]
        sgd[nb:nb + nr] = R.sig[iR] if nr else sgd[nb:nb + nr]
        Dr = np.zeros(n); Dr[:nb] = B.Drad[iB]
        if nr:
            Dr[nb:nb + nr] = R.Drad[iR]
        aV = np.zeros(n); aV[:nb] = B.R[iB]
        if nr:
            aV[nb:nb + nr] = R.R[iR]
        if vent is not None and mm == 0:
            aV[isV > 0] = -vent[1]

        roll = np.exp(-0.5 * (B.k[iB] * tip) ** 2)
        cf = _contact_filter(B.om[iB], tau)
        ph_s = _phi(B, iB, FR[p.s], p.th)
        strike = np.zeros(n); strike[:nb] = ph_s * roll * cf
        mis = np.zeros(n); mis[:nb] = ph_s
        kd = np.zeros(n); kd[:nb] = B.k[iB] ** 2 * B.norm[iB]
        tf = np.zeros(n); tf[:nb] = B.omT2[iB] / (B.omT2[iB] + B.omD2[iB])
        if p.mic:
            orad = Dr * (isB + MIC_RESO * isR)
            if vent is not None and mm == 0:
                orad = orad + MIC_PORT * vent[1] * isV
        else:
            orad = np.zeros(n)
            orad[:nb] = _phi(B, iB, FR[PICKUP_FR], PICKUP_TH)

        om = np.sqrt(np.maximum(lam, 0))
        dM = np.diag(M)
        X2 = X * X
        ke = dM[:, None] * X2
        ws = ke.sum(0)
        if mm == 0:
            vol = aV @ X
            rad = RAD_C * om ** 2 * vol ** 2 / B.smu
        else:
            rad = RAD_C * om ** 2 * (((Dr * isB) @ X) ** 2 + ((Dr * isR) @ X) ** 2)
        sig = (sgd @ ke) / ws + rad
        out = orad @ X
        if p.mic:
            out = -out * om ** 2
        tsh = ((tf * isB) @ ke) / ws

        C['om'].append(om); C['sig'].append(sig); C['out'].append(out)
        C['v'].append(strike @ X); C['mi'].append(mis @ X)
        C['kern'].append(kd @ X2)
        C['oT'].append(lam * tsh); C['oD'].append(lam * (1 - tsh))
        C['partB'].append((isB @ ke) / ws); C['vfrac'].append((isV @ ke) / ws)
        C['m'].append(np.full(len(om), mm))
        if wires_on:
            ph = np.zeros(wr.shape + (n,))
            if nr:
                ph[..., nb:nb + nr] = _phi(R, iR, wr, wth)
            C['shape'].append(ph.reshape(-1, n) @ X)
    Cc = {k: (np.concatenate(v, axis=-1) if v else np.zeros(0)) for k, v in C.items()}
    nC = len(Cc['om'])

    # ---- U: struck-sheet modes the cavity cannot reach --------------------
    U = np.flatnonzero(~inB)
    S = B2
    roll = np.exp(-0.5 * (S.k[U] * tip) ** 2)
    cf = _contact_filter(S.om[U], tau)
    miU = _phi(S, U, FR[p.s], p.th)
    omU = S.om[U]
    sigU = S.sig[U] + RAD_C * omU ** 2 * S.Drad[U] ** 2 / S.mass[U]
    outU = -S.Drad[U] * omU ** 2 if p.mic else _phi(S, U, FR[PICKUP_FR], PICKUP_TH)
    vU = miU * roll * cf / S.mass[U]
    kernU = S.k[U] ** 2 * S.norm[U]

    # ---- W: far-sheet modes only the wires reach ---------------------------
    if wires_on:
        Wi = np.flatnonzero(~inR & (R2.f < F_WIRE))
        sq = np.sqrt(R2.mass[Wi])
        omW = R2.om[Wi]
        sigW = R2.sig[Wi] + RAD_C * omW ** 2 * R2.Drad[Wi] ** 2 / R2.mass[Wi]
        outW = (-MIC_RESO * R2.Drad[Wi] * omW ** 2 / sq if p.mic else np.zeros(len(Wi)))
        shW = _phi(R2, Wi, wr, wth).reshape(-1, len(Wi)) / sq
    else:
        Wi = np.array([], int)
        omW = sigW = outW = np.zeros(0)
    nW = len(Wi)

    # ---- merge, then order by LIFETIME --------------------------------------
    # With constant Q a 10 kHz mode is inaudible after ~25 ms while the
    # fundamental runs a second, so most modes are dead for most of the render.
    # Sorting by death sample lets the loop shrink its active set rather than
    # grinding zeros -- about 15x, and exact.  Anything the wires can touch is
    # held alive for the whole render: a dead mode can be struck again.
    nU = len(U)
    om = np.concatenate([omU, Cc['om'], omW])
    sig = np.concatenate([sigU, Cc['sig'], sigW])
    out = np.concatenate([outU, Cc['out'], outW])
    v = np.concatenate([vU, Cc['v'], np.zeros(nW)])
    kn = np.concatenate([kernU, Cc['kern'], np.zeros(nW)])
    mif = np.concatenate([miU, Cc['mi'], np.zeros(nW)])
    oT = np.concatenate([S.omT2[U], Cc['oT'], np.zeros(nW)])
    oD = np.concatenate([S.omD2[U], Cc['oD'], omW ** 2])
    life = np.minimum(DEAD_DB / 20 * math.log(10) / np.maximum(sig, 1e-6), DUR) * SR
    life_nat = life.copy()
    touch = np.zeros(len(om), bool)
    if wires_on:
        shC = Cc['shape'] if nC else np.zeros((wr.size, 0))
        wsh_all = np.concatenate([np.zeros((wr.size, nU)), shC, shW], axis=1)
        touch = np.abs(wsh_all).max(0) > 0
        life[touch] = DUR * SR + 1
    o = np.argsort(-life, kind='stable')
    om, sig, out, v, kn, mif, life, oT, oD, touch, life_nat = (
        x[o] for x in (om, sig, out, v, kn, mif, life, oT, oD, touch, life_nat))

    widx = wshape = wire = None
    if wires_on:
        wsh_all = wsh_all[:, o]
        widx = np.flatnonzero(touch)
        wshape = wsh_all[:, widx]
        mu = p.wMu * 1e-3
        y0 = W.rest_shape(wu, whalf, p.wBed * 1e-3)
        wire = dict(n=int(p.wN), pts=wr.shape[1], life=life_nat[widx],
                    T=np.full(len(wls), float(p.wT)), mu=mu, ls=wls, m=mu * wls, y0=y0,
                    half=whalf, press=p.wBed * 1e-3,
                    f_end=p.wT * max(p.wBed, 0.0) * 1e-3 / wls,
                    r=wr, th=wth)

    beta = B.mat.E * B.h / (2 * np.pi * p.a * p.a * p.T)
    tauL = 2 * p.a / math.sqrt(B.mat.E / B.mat.rho)

    fC_all = Cc['om'] / (2 * np.pi) if nC else np.zeros(0)
    cand = fC_all[(fC_all > 20) & (Cc['partB'] > 0.30)] if nC else np.zeros(0)
    f1 = float(cand.min()) if len(cand) else 1e9
    if nU and omU.min() / (2 * np.pi) < f1:
        f1 = float(omU.min() / (2 * np.pi))
    iF1 = int(np.argmin(np.abs(om / (2 * np.pi) - f1)))

    m0 = (Cc['m'] == 0) if nC else np.zeros(0, bool)
    fC = fC_all[m0]
    pB0 = Cc['partB'][m0] if nC else np.zeros(0)
    f_vent = 0.0
    if vented:
        q = fC[(Cc['vfrac'][m0] > 0.3) & (fC > 5)]
        f_vent = float(q.min()) if len(q) else 0.0
    f_air = (0.0, 0.0)
    if two:
        q = np.sort(fC[(pB0 > 0.15) & (pB0 < 0.99) & (fC > 20)])
        if len(q) > 1:
            f_air = (float(q[0]), float(q[1]))

    info = dict(material=B.mat.name, n_modes=B.n, f_top=float(B.f.max()),
                stiffness=B.stiffness_share, air_fraction=float(B.se[0] / B.smu - 1),
                tip_eff=tip, tip_limited=bool(tip_limited), two_sheet=two,
                vented=bool(vented), f_vent=f_vent, f_air=f_air,
                volume=math.pi * p.a * p.a * L, c=B.c, sheet=B, reso=R,
                j_cav=jcav, f_dyn=f_dyn, n_blocks=len(orders), wires=bool(wires_on),
                c_m=Cc['m'] if nC else np.zeros(0), c_f=fC_all, c_partB=Cc['partB'] if nC else np.zeros(0),
                bed_force=float(wire['f_end'].mean()) if wires_on else 0.0,
                strands=int(wire['n']) if wires_on else 0)

    return Drum(om=om, sig=sig, out=out, v=v, kern=kn, mi=mif, life=life,
                omT2=oT, omD2=oD, beta=beta, tauL=tauL, tau=tau, iF1=iF1,
                f1=f1 if np.isfinite(f1) else 0.0, fC=fC,
                nU=nU, nC=nC, nW=nW, widx=widx, wshape=wshape, wire=wire, info=info)


# ------------------------------------------------------------------- render
def render(d, impulse=None, vel=1.0, dur=DUR, sr=SR, limiter=True, trace=False):
    """One strike.  `impulse` in N.s.

    State is (q, v) -- position and velocity -- advanced by the exact
    damped-oscillator rotation.  The two-pole form q[n+1] = a1 q[n] - a2 q[n-1]
    is exact at a FIXED frequency, but tension modulation changes the frequency
    every few samples, and swapping a1 while keeping both past samples is not
    energy-consistent.  Position and velocity are physical, so the same rotation
    with a new omega is simply the same state seen by a different oscillator.

    A force impulse J applied just before a step moves the step's result by
    (sw J, (cc - sigma sw) J) -- that is how the wires push on the modes, and it
    is exact for the rotation, not an Euler approximation.
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
    v = np.zeros(n)
    dc = 0.0
    frozen = False

    wz = d.wire
    if wz is not None:
        wi, Sh = d.widx, d.wshape
        m_u = wz['m'][:, None] * np.ones((1, wz['pts']))
        y0 = wz['y0']
        hs = (dt * dt / m_u).ravel()
        # The one-step compliance between every pair of points: sheet plus
        # strand.  The contact forces are solved against all of it at once.
        Cmat = dt * (Sh * sw[wi]) @ Sh.T
        A = Cmat + np.diag(hs)
        # Start at rest WITH the preload on.  Each point bears its strand's
        # elastic force, the sheet sits at the discrete fixed point of the
        # integrator under that constant push, and each strand sits at the
        # small overlap the contact law needs to hold exactly that force.
        # Starting from zero would fire the preload as a step: a thump that is
        # not in the drum.
        yw = np.zeros_like(y0)
        vw = np.zeros_like(y0)
        # Fixed point of the integrator under a constant kick D per step:
        #   q' = c q + s(v + sg q) + s D,   v' = c v - s(w2 q + sg v) + (c - sg s) D
        # which is linear, q* = kq D and v* = kv D.
        c_, s_, sg_ = cc[wi], sw[wi], d.sig[wi]
        a11, a12 = 1 - c_ - s_ * sg_, -s_
        a21, a22 = s_ * w2[wi], 1 - c_ + s_ * sg_
        det = a11 * a22 - a12 * a21
        kq = (s_ * a22 - a12 * (c_ - sg_ * s_)) / det
        kv = (a11 * (c_ - sg_ * s_) - a21 * s_) / det
        # Strand position y = G F0 (the sheet's deflection -- the contact is
        # rigid, so no overlap), and the strand's own force F0 = E(0) + Lw y.  The
        # sheet gives way under the strands by a fair fraction of the bed depth,
        # so this is solved, not iterated: a plain fixed point oscillated with
        # ratio -0.85 and had not settled in 4 rounds.
        # With no bed depth (or a gap) nothing presses at rest: the strands hang at
        # their rest shape and the sheet is undisturbed.  A contact can only
        # push, so that case must not go through the solve, which would happily
        # return negative forces and pull the sheet out to meet the strands.
        if wz['press'] > 0:
            G = dt * (Sh * kq) @ Sh.T
            Lw = W.elastic_matrix(wz['T'], wz['ls'], wz['pts'])
            E0 = W.elastic(yw, y0, wz['T'], wz['ls']).ravel()
            F0 = np.linalg.solve(np.eye(len(E0)) - Lw @ G, E0)
            F0 = np.maximum(F0, 0.0)
            D = dt * (Sh.T @ F0)
            q[wi] = kq * D
            v[wi] = kv * D
            v_rest = v[wi].copy()
            yw = (G @ F0).reshape(y0.shape)
        else:
            yw = y0.copy()
            v_rest = np.zeros(len(wi))
        # The microphone reads -omega^2 q as acceleration, which is only true of
        # motion.  The preload's static dent is not sound: take it out.
        dc = float(d.out[wi] @ q[wi])
        n_pts = y0.size
        cact = None
        frozen, t_frozen, e_pk = False, N, 0.0
        n_touch = 0
        n_sep = 0
        fpk = 0.0
        if trace:
            tr = dict(F=np.zeros(N), sep=np.zeros(N), w=np.zeros((n_pts, N)), E=np.zeros(N))
            Tl = wz['T'] / wz['ls']
    v += P * d.v                                  # the strike is a velocity kick

    a_lag = 1 - math.exp(-dt / max(d.tauL, dt))
    modulatable = d.om <= OM_MOD

    g, gmax, dent, act, clipped, Ss = 1.0, 1.0, 0.0, n, False, 0.0
    n_fast = int(0.001 * sr)
    for t in range(1, N):
        while act > 0 and d.life[act - 1] <= t:
            act -= 1
        sl = slice(0, act)
        # qi must be a COPY: q[sl] is a view, so writing q[sl] first would feed
        # the already-updated position into the velocity line.
        if frozen:
            q_pre, v_pre = q[wi].copy(), v[wi].copy()
        qi = q[sl].copy()
        vi = v[sl]
        sg = d.sig[sl]
        q[sl] = cc[sl] * qi + sw[sl] * (vi + sg * qi)
        v[sl] = cc[sl] * vi - sw[sl] * (w2[sl] * qi + sg * vi)

        if wz is not None and not frozen:
            E = W.elastic(yw, y0, wz['T'], wz['ls'])
            vwf = vw + dt * (E / m_u - 2 * WIRE_SIG * vw)
            ywf = yw + dt * vwf
            wf = Sh @ q[wi]
            eta = ywf.ravel() - wf
            F, cact = W.solve_contact(A, eta, cact)
            J = dt * (Sh.T @ F)
            q[wi] += sw[wi] * J
            v[wi] += (cc[wi] - d.sig[wi] * sw[wi]) * J
            Fm = F.reshape(y0.shape)
            yw = ywf - dt * dt * Fm / m_u
            vw = vwf - dt * Fm / m_u
            tc = F > 0
            n_touch += int(tc.sum())
            n_sep += n_pts - int(tc.sum())
            fpk = max(fpk, float(F.max()) if n_pts else 0.0)
            if trace:
                tr['w'][:, t] = Sh @ q[wi]
                tr['F'][t] = F.sum()
                tr['sep'][t] = 1 - tc.mean()
                # energy of everything the contact couples: the members the
                # wires touch (unit modal mass) and the strands -- kinetic plus
                # the tension energy of each segment, anchors included.  With
                # every loss off this must never rise (tests/test_physics.py).
                dd = yw - y0
                z = np.zeros((dd.shape[0], 1))
                seg = np.diff(np.concatenate([z, dd, z], axis=1), axis=1)
                tr['E'][t] = 0.5 * float(v[wi] @ v[wi] + (w2[wi] * q[wi]) @ q[wi]
                                         + (m_u * vw * vw).sum()
                                         + (Tl[:, None] * seg * seg).sum())
            # Once the sheet under the strands has fallen WIRE_QUIET below its
            # peak nothing the wires do can be heard, and they are left resting
            # where they are: the last forces stay on as a constant push, which
            # is the preload they would settle to anyway.
            if t % 441 == 0:
                # acceleration, not velocity: a 42 Hz vent mode moves the sheet
                # for a second and could not throw a strand if it tried
                dv = (v[wi] - v_rest) * d.om[wi]
                ek = float(dv @ dv)
                e_pk = max(e_pk, ek)
                if ek < e_pk * 10 ** (-WIRE_QUIET / 10):
                    frozen, t_frozen = True, t
                    J_rest = dt * (Sh.T @ F)
        elif wz is not None:
            # Frozen: the strands are a constant push, and each member the wires
            # held alive goes back to its own lifetime.  One that has decayed
            # DEAD_DB is sitting at its static deflection -- already taken out
            # as dc -- so it is simply no longer advanced: undo this step's
            # rotation for it, and give it no push.
            dead = wz['life'] <= t
            q[wi[dead]] = q_pre[dead]
            v[wi[dead]] = v_pre[dead]
            live = ~dead
            q[wi[live]] += sw[wi[live]] * J_rest[live]
            v[wi[live]] += (cc[wi[live]] - d.sig[wi[live]] * sw[wi[live]]) * J_rest[live]

        qn = q[sl]
        y[t] = float(d.out[sl] @ qn) - dc
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

    raw = y.copy() if trace else None
    pk = float(np.abs(y).max())
    if pk > 0:
        y *= 0.92 / pk
    drive = 1.0
    if limiter:
        Wn = min(int(RMS_WIN * sr), N)
        rms = float(np.sqrt((y[:Wn] ** 2).mean()))
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
    rep = dict(gmax=gmax, bend=bend, dent=dent, drive=drive, clipped=clipped)
    if wz is not None:
        rep.update(sep=n_sep / max(n_sep + n_touch, 1), fpk=fpk, t_frozen=t_frozen / sr)
        if trace:
            rep.update(trace=tr, raw=raw)
    elif trace:
        rep.update(raw=raw)
    return y, rep
