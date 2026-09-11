"""
THE OLD ENGINE.  Not the one in physicaldrum/ -- do not build on this.

This is the pre-stiffness Python engine: air loading and constant-Q damping,
but no bending stiffness, no materials beyond Mylar, no in-plane tension lag,
no (q,v) integrator.  It is kept only because snare_wires.py was written
against it and has not been carried across.  Everything it does, the current
engine does better; the one thing it has that the current engine does not is
the per-sample forcing hook the wire contact needs.
"""

import json, numpy as np
from scipy.special import jv
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physicaldrum.air import added_mass_coeff as A_coeff
from physicaldrum.tables import air_lookup as _air_lookup
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))

def loaded(k, m, j, a, T, smu, sides=1, iters=14, idx=None):
    """Old self-consistent air loading, kept for the experiments only."""
    import numpy as _np
    from physicaldrum.constants import RHO_AIR as _RA, C_AIR as _CA
    se = _np.full_like(k, smu)
    for _ in range(iters):
        c = _np.sqrt(T/se)
        A = _air_lookup(idx, c/_CA) if idx is not None else A_coeff(m, j, c/_CA)
        se = 0.35*se + 0.65*(smu + sides*_RA*a*A)
    return se, k*_np.sqrt(T/se)

def _p(name): return _os.path.join(_HERE, name)

D = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'modes_full.json')))
TM  = np.array(D['M']);  TJ = np.array(D['J']); TJP = np.array(D['JP'])
TPHI = np.array(D['PHI']); TFR = np.array(D['FR'])

RHO_A, C_A = 1.204, 343.0
K_AIR_NUM  = RHO_A * C_A**2


def head(a, T, h_mil, s0, eta, fmax=6000, maxm=260, rho=1390.0, airload=True):
    h   = h_mil * 2.54e-5
    smu = rho * h
    jmax = 2*np.pi*a*fmax/np.sqrt(T/smu)          # dry band edge (wet is lower)
    idx = np.where(TJ < jmax)[0][:maxm]
    m, j, Jp = TM[idx], TJ[idx], TJP[idx]
    k = j/a

    # ---- A. finite-disc air added mass (see airload.py).  ONE open side:
    #      the head faces the room on one side; the other faces the sealed
    #      cavity, whose air is a COMPLIANCE (the K_air rank-1 term), not an
    #      unbounded half space.  Validated against measured timpani partial
    #      ratios to 3.7% (an ideal membrane is off by 21%).
    if airload:
        se, om = loaded(k, m, j, a, T, smu, sides=1)
    else:
        se = np.full_like(k, smu); om = k*np.sqrt(T/se)
    f  = om/(2*np.pi)

    eps  = np.where(m == 0, 2.0, 1.0)
    norm = eps*(np.pi*a*a/2)*Jp*Jp                # INT INT Phi^2 dA
    R    = np.where(m == 0, 2*np.pi*a*a*(-Jp)/j, 0.0)
    sig  = s0 + eta*om/2                          # ---- B. constant loss factor
    return dict(idx=idx, m=m, j=j, Jp=Jp, k=k, f=f, om=om, norm=norm,
                R=R, sig=sig, smu=smu, se=se, a=a, T=T, h=h,
                mass=se*norm, airfrac=se/smu - 1)


def directivity(hd, theta, phi=0.0):
    m, j, Jp, om, a = hd['m'], hd['j'], hd['Jp'], hd['om'], hd['a']
    beta = (om*a/C_A)*np.sin(theta)
    return 2*np.pi*a*a*j*Jp*jv(m, beta)/(beta*beta - j*j)*np.cos(m*phi)


def secular(om2, w2, K):
    o = np.argsort(om2); om2 = om2[o].copy(); w2 = w2[o]
    for t in range(1, len(om2)):
        om2[t] = max(om2[t], om2[t-1]*(1+1e-7)+1e-6)
    n = len(om2); lam = np.zeros(n)
    g = lambda l: 1.0 + K*np.sum(w2/(om2 - l))
    for i in range(n):
        lo = om2[i]; hi = om2[i+1] if i < n-1 else om2[-1]+K*np.sum(w2)+1.0
        a_ = lo + 1e-9*max(1.0, abs(lo)); b_ = hi - 1e-9*max(1.0, abs(hi))
        for _ in range(120):
            mid = 0.5*(a_+b_)
            if g(mid) > 0: b_ = mid
            else:          a_ = mid
        lam[i] = 0.5*(a_+b_)
    return lam, o, om2


def build(batter, reso=None, depth=0.14, port_r=0.0, port_len_fac=1.7,
          theta=1.05, phi=0.0, mic_reso=0.30, mic_port=0.75, rad_c=3.0e-5):
    b = batter
    ub = b['m'] >= 1
    Db = directivity(b, theta, phi)
    om2=[]; mass=[]; aV=[]; oR=[]; sg=[]; src=[]
    for i in np.where(b['m'] == 0)[0]:
        om2.append(b['om'][i]**2); mass.append(b['mass'][i]); aV.append(b['R'][i])
        oR.append(Db[i]); sg.append(b['sig'][i]); src.append(('B', i))
    if reso is not None:
        r = reso; Dr = directivity(r, theta, phi)
        for i in np.where(r['m'] == 0)[0]:
            om2.append(r['om'][i]**2); mass.append(r['mass'][i]); aV.append(r['R'][i])
            oR.append(mic_reso*Dr[i]); sg.append(r['sig'][i]); src.append(('R', i))
    V = np.pi*b['a']**2*depth
    if port_r > 0:
        Ap = np.pi*port_r**2; Leff = port_len_fac*port_r
        om2.append(0.0); mass.append(RHO_A*Ap*Leff)
        aV.append(-Ap); oR.append(mic_port*Ap); sg.append(8.0); src.append(('P', 0))
    om2=np.array(om2); mass=np.array(mass); aV=np.array(aV); oR=np.array(oR); sg=np.array(sg)
    K_air = K_AIR_NUM/V
    w = aV/np.sqrt(mass)
    lam, order, o2s = secular(om2, w*w, K_air)
    ws = w[order]
    X = np.zeros((len(om2), len(lam)))
    for c_ in range(len(lam)):
        z = ws/(o2s - lam[c_]); z /= np.linalg.norm(z); X[:, c_] = z
    Xp = X/np.sqrt(mass[order])[:, None]
    outC = (oR[order][:, None]*Xp).sum(0)
    volC = (aV[order][:, None]*Xp).sum(0)
    sigC = (X*X*sg[order][:, None]).sum(0)
    partB = (X*X*np.array([[1.0 if src[i][0]=='B' else 0.0] for i in order])).sum(0)
    return dict(lam=lam, f=np.sqrt(np.maximum(lam,0))/(2*np.pi), mass=np.ones(len(lam)),
                out=outC, vol=volC, sig=sigC, X=Xp, order=order, partB=partB,
                src=[src[i] for i in order], K_air=K_air, V=V,
                b=b, ub=ub, Db=Db, rad_c=rad_c, theta=theta)


def render(cfg, P=1.7e-2, i_s=8, w_stick=0.010, tau=0.5e-3, vel=1.0,
           dur=1.6, sr=44100, E=4.0e9, tension=True):
    b, ub = cfg['b'], cfg['ub']
    def cfilt(om):
        if tau <= 0: return np.ones_like(om)
        x = om*tau/np.pi
        return np.clip(np.abs(np.sinc(om*tau/(2*np.pi))/(1-x*x+1e-9)), 0, 1)
    om  = b['om'][ub]; sigU = b['sig'][ub].copy()
    mi  = TPHI[i_s][b['idx']][ub]
    seU = b['se'][ub]; nrmU = b['norm'][ub]
    roll = np.exp(-0.5*(b['k'][ub]*w_stick)**2)
    vU  = P*vel*mi*roll*cfilt(om)/(seU*nrmU)
    outU = -cfg['Db'][ub]*om**2
    sigU += cfg['rad_c']*om**2*cfg['Db'][ub]**2/(seU*nrmU)
    kern = b['k'][ub]**2*nrmU
    omC  = np.sqrt(np.maximum(cfg['lam'], 0))
    sigC = cfg['sig'] + cfg['rad_c']*omC**2*cfg['vol']**2/b['smu']
    outC = -cfg['out']*omC**2
    Fb = np.zeros(len(cfg['src']))
    for r_, s_ in enumerate(cfg['src']):
        if s_[0] == 'B':
            i = s_[1]
            Fb[r_] = (P*vel*TPHI[i_s][b['idx'][i]]
                      * np.exp(-0.5*(b['k'][i]*w_stick)**2)
                      * float(cfilt(np.array([b['om'][i]]))[0]))
    vC = (cfg['X']*Fb[:, None]).sum(0)
    om_a=np.concatenate([om,omC]); sig_a=np.concatenate([sigU,sigC])
    v_a=np.concatenate([vU,vC]);   out_a=np.concatenate([outU,outC])
    nU=len(om)
    dt=1/sr; n=int(dur*sr)
    decay=np.exp(-sig_a*dt); a2=np.exp(-2*sig_a*dt)
    wd=np.sqrt(np.maximum(om_a**2-sig_a**2,1e-9)); a1=2*decay*np.cos(wd*dt)
    beta=E*b['h']/(2*np.pi*b['a']**2*b['T'])
    qp=np.zeros_like(om_a); qc=np.zeros_like(om_a); y=np.zeros(n); g=1.0; gmax=1.0
    for i in range(1,n):
        qn=a1*qc-a2*qp; qp=qc; qc=qn
        if i==1: qc=qc+(v_a/wd)*decay*np.sin(wd*dt)
        y[i]=out_a@qc
        if tension and i % (4 if i<44 else 8)==0:
            S=float(kern@(qc[:nU]**2)); gi=min(np.sqrt(1+beta*S),2.0); gmax=max(gmax,gi)
            if abs(gi-g)>1e-6:
                g=gi; wd=np.sqrt(np.maximum((g*om_a)**2-sig_a**2,1e-9)); a1=2*decay*np.cos(wd*dt)
    return y, gmax
