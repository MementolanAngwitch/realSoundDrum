"""
Snare wires: a one-sided contact against the resonant head.

Why this needs new machinery.  Everywhere else in the engine a strike sets
initial velocities and the modes then run free -- linear, closed form, no
per-sample forcing.  Snares are not that.  A wire rests on the underside of the
reso head under a preload.  When the head accelerates away faster than the
preload can follow the wire LOSES CONTACT, flies free, and comes back and hits
it.  The snap is that train of collisions, and a collision is a force that
exists only while the two bodies overlap -- a genuine per-sample nonlinearity.

Consequences for the model:
  * the reso head's m>=1 modes come back.  They were dropped because the cavity
    cannot drive them (their net volume is zero), which was true -- but the
    wires drive them directly, at a point, and they are where the broadband
    content lives.
  * every mode now needs a forcing term, not just an initial condition.  For the
    exact two-pole recurrence the impulse-invariant input is
        q[n+1] += (f[n]/m_i) . e^{-sigma dt} sin(w_d dt)/w_d
    which is the same factor the strike already uses.

Contact law: one-sided linear spring with contact damping,
    f = K_c [eta]_+ + C_c [eta_dot]_+ ,   eta = w_head - z_wire + preload
The one-sidedness IS the physics; the spring constant is set from a target
contact time tau_c = pi sqrt(m_w/K_c), which is what fixes the brightness --
a shorter collision is a wider spectrum.
"""
import numpy as np
from _legacy_engine import head, build, TPHI, TFR, directivity, C_A
from scipy.special import jv


def phi_at(hd, r_frac, th):
    """mode shape of `hd` at radius r_frac*a, angle th"""
    return jv(hd['m'], hd['j']*r_frac)*np.cos(hd['m']*th)


def render_snared(cfg, reso, P=2.0e-2, i_s=12, w_stick=0.006, tau=0.35e-3, vel=1.0,
                  dur=0.9, sr=44100, E=4.0e9, tension=True,
                  n_pts=16, n_per=1, m_w=2.2e-5, f_w=(1300., 2700.),
                  tau_c=70e-6, tc_spread=0.45, F0_n=0.020, c_c=0.30, wire_out=2.5e-4,
                  span=0.80, sig_w=26.0, mic_reso=0.30, seed=3):
    b, ub = cfg['b'], cfg['ub']
    rng = np.random.default_rng(seed)
    def cfilt(om):
        if tau <= 0: return np.ones_like(om)
        x = om*tau/np.pi
        return np.clip(np.abs(np.sinc(om*tau/(2*np.pi))/(1-x*x+1e-9)), 0, 1)

    # ---------- A. batter m>=1 : struck, free ------------------------------
    omA=b['om'][ub]; sigA=b['sig'][ub].copy()
    seA=b['se'][ub]; nrA=b['norm'][ub]
    rollA=np.exp(-0.5*(b['k'][ub]*w_stick)**2)
    vA=P*vel*TPHI[i_s][b['idx']][ub]*rollA*cfilt(omA)/(seA*nrA)
    outA=-cfg['Db'][ub]*omA**2
    sigA+=cfg['rad_c']*omA**2*cfg['Db'][ub]**2/(seA*nrA)
    kern=b['k'][ub]**2*nrA

    # ---------- B. coupled normal modes : struck, and forced by the wires ---
    omB=np.sqrt(np.maximum(cfg['lam'],0))
    sigB=cfg['sig']+cfg['rad_c']*omB**2*cfg['vol']**2/b['smu']
    outB=-cfg['out']*omB**2
    Fb=np.zeros(len(cfg['src']))
    for r_,s_ in enumerate(cfg['src']):
        if s_[0]=='B':
            i=s_[1]
            Fb[r_]=(P*vel*TPHI[i_s][b['idx'][i]]
                    *np.exp(-0.5*(b['k'][i]*w_stick)**2)
                    *float(cfilt(np.array([b['om'][i]]))[0]))
    vB=(cfg['X']*Fb[:,None]).sum(0)

    # ---------- C. reso m>=1 : silent until the wires hit them --------------
    ur=reso['m']>=1
    omC=reso['om'][ur]; sigC=reso['sig'][ur].copy()
    seC=reso['se'][ur]; nrC=reso['norm'][ur]
    Dr=directivity(reso,cfg['theta'])[ur]
    outC=-mic_reso*Dr*omC**2
    sigC+=cfg['rad_c']*omC**2*Dr**2/(seC*nrC)
    massC=seC*nrC

    # ---------- D. the wires -----------------------------------------------
    # contact points spread along a chord, plus n_per wire units at each
    rp=np.linspace(-span,span,n_pts)
    r_f=np.abs(rp); th=np.where(rp<0,np.pi,0.0)
    PhiC=np.array([phi_at({'m':reso['m'][ur],'j':reso['j'][ur]},r_f[i],th[i])
                   for i in range(n_pts)])                    # (n_pts, nC)
    # reso m=0 rows inside the coupled block -> their shape at each point
    m0=[t for t,s_ in enumerate(cfg['src']) if s_[0]=='R']
    if m0:
        j0=np.array([reso['j'][cfg['src'][t][1]] for t in m0])
        PhiB=np.array([jv(0,j0*r_f[i]) for i in range(n_pts)])  # (n_pts, len(m0))
        XpR=cfg['X'][m0,:]                                      # (len(m0), nB)
        PhiB_n=PhiB@XpR                                         # (n_pts, nB)
    else:
        PhiB_n=np.zeros((n_pts,len(omB)))

    # ---------- D. the wires ------------------------------------------------
    # Geometry, positive along the drum axis pointing AWAY from the batter.
    # The reso head's surface at a contact point is w; the wire sits just
    # outside it at z, held against it by its own tension with a static preload
    # force F0.  They can only push, never pull:
    #     eta = w - z          penetration, contact only while eta > 0
    #     f   = [Kc eta - Cc zdot]_+
    #     m zddot = f - k_w (z + delta) - c zdot,      k_w delta = F0
    # The wire leaves the head when the head retreats faster than F0/m can pull
    # the wire after it.  F0/m is therefore THE parameter that decides whether a
    # snare buzzes at all -- it is the throw-off tension, and it competes
    # directly against the head's peak acceleration.
    NW=n_pts
    fw=rng.uniform(f_w[0],f_w[1],NW)
    mw=m_w*rng.uniform(0.75,1.3,NW)
    omw=2*np.pi*fw
    kw=mw*omw**2
    F0=F0_n*rng.uniform(0.65,1.45,NW)          # static preload per unit, N
    delta=F0/kw
    # Contact time varies wire to wire (different mass, different local head
    # compliance).  Spreading it matters: with one shared tau_c every wire has
    # the same contact resonance and they sing together instead of hissing.
    tcs=tau_c*rng.uniform(1-tc_spread,1+tc_spread,NW)
    Kc=np.pi**2*mw/tcs**2                      # contact stiffness from contact TIME
    Cc=c_c*2*np.sqrt(Kc*mw)

    # ---------- run ---------------------------------------------------------
    dt=1/sr; n=int(dur*sr)
    om=np.concatenate([omA,omB,omC]); sig=np.concatenate([sigA,sigB,sigC])
    out=np.concatenate([outA,outB,outC]); v0=np.concatenate([vA,vB,np.zeros(len(omC))])
    nA,nB,nC=len(omA),len(omB),len(omC)
    dec=np.exp(-sig*dt); a2=np.exp(-2*sig*dt)
    wd=np.sqrt(np.maximum(om**2-sig**2,1e-9)); a1=2*dec*np.cos(wd*dt)
    gin=dec*np.sin(wd*dt)/wd
    # The strike enters as an IMPULSE (N.s), so q += (impulse/m)*gin.  A contact
    # force is N, so its impulse over one sample is f*dt -- that dt is not
    # optional: without it the injection and the implicit self-term hpp are both
    # wrong by 1/dt (~44100), which silently crushes every collision to nothing.
    ginB=gin[nA:nA+nB]*dt; ginC=gin[nA+nB:]*dt/massC
    beta=E*b['h']/(2*np.pi*b['a']**2*b['T'])

    # Implicit self-term.  The contact is stiff; coupling it explicitly (head
    # pushes wire, wire pushes head, both from last sample) blows up.
    hpp=(PhiB_n*PhiB_n)@ginB + (PhiC*PhiC)@ginC

    qp=np.zeros_like(om); qc=np.zeros_like(om); y=np.zeros(n)
    zw=-F0/(Kc+kw); vw=np.zeros(NW)            # static equilibrium, resting
    g=1.0; gmax=1.0; hits=0; nsep=0; fpk=0.0
    cw=2*0.05*np.sqrt(kw*mw)
    for i in range(1,n):
        qn=a1*qc-a2*qp; qp=qc; qc=qn
        if i==1: qc=qc+v0*gin
        w_pt=PhiB_n@qc[nA:nA+nB] + PhiC@qc[nA+nB:]      # free surface
        num=Kc*(w_pt-zw)-Cc*vw
        f=np.maximum(num/(1.0+Kc*hpp),0.0)              # one-sided, implicit
        touching=f>0
        hits+=int(touching.sum()); nsep+=int((~touching).sum())
        fp=float(np.abs(f).max())
        if fp>fpk: fpk=fp
        vw=vw+((f-kw*(zw+delta)-cw*vw)/mw)*dt
        zw=zw+vw*dt
        qc[nA:nA+nB]+=ginB*(PhiB_n.T@(-f))
        qc[nA+nB:]  +=ginC*(PhiC.T@(-f))
        y[i]=out@qc + wire_out*f.sum()
        if tension and i%(4 if i<44 else 8)==0:
            S=float(kern@(qc[:nA]**2)); gi=min(np.sqrt(1+beta*S),2.0); gmax=max(gmax,gi)
            if abs(gi-g)>1e-6:
                g=gi; wd=np.sqrt(np.maximum((g*om)**2-sig**2,1e-9)); a1=2*dec*np.cos(wd*dt)
    return y,gmax,dict(sep=nsep/(n*NW),fpk=fpk,accmax=F0_n/m_w)
