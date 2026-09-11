"""
The cavity, done with its real mode shapes instead of one lumped compliance.

Why.  Treating the enclosed air as a single uniform pressure makes the coupling
proportional to INT INT Phi dA, which is exactly zero for every m>=1 mode.  So
in that model the batter head CANNOT drive the reso head at any angular order
above zero -- the reso head only ever sees a slow breathing motion.  That is
fine up to the first non-uniform cavity mode, 1.8412 c/(2 pi a) = 565 Hz for a
14" snare, and wrong above it.  It is also why the snare wires had nothing to
rattle against: they were resting on a 292 Hz sine.

The real cavity modes of a rigid cylinder with two flexible ends are
    Psi_mqs = J_m(chi_mq r/a) cos(m theta) cos(s pi x/L),  chi_mq = q-th zero of J_m'
and a head mode Phi_mn couples only to cavity modes of the SAME m, so the whole
thing stays block diagonal in angular order.  The overlap is Lommel again --
the same closed form as the far field and the added mass, for the same reason
(J_m(j_mn) = 0 at the clamped rim):

    C = eps pi a^2 . (-j J_m'(j) J_m(chi))/(j^2 - chi^2)

chi = 0, m = 0 recovers INT INT Phi dA, so the old rank-1 model is the q=s=0
term of this one.

STATUS — UNFINISHED, AND ITS EIGENVECTORS ARE NOT TRUSTWORTHY.

The eigenvalues are right: restricted to the single uniform cavity mode this
reproduces the rank-one model's coupled tom pair to 0.1 Hz, which is a real
check.  But `eig(B, A)` here is a NON-SYMMETRIC generalized problem, so the
returned vectors are not mass-orthonormal and cannot be used for modal
participation as they stand — a strike-projection diagnostic built on them
reported ~0 batter->reso transmission at every angular order, including m = 0
where the rank-one model demonstrably does transmit.

Do not use this to drive audio until either the left eigenvectors are taken
properly, or the system is recast in a symmetric form (an air displacement
potential formulation would do it, and would also make a JS port tractable —
there is no dense eigensolver in the browser build).
"""
import numpy as np
from scipy.special import jv, jvp, jnp_zeros
from scipy.linalg import eig
RHO_A, C_A = 1.204, 343.0

def chi_table(mmax, qmax):
    """zeros of J_m' (rigid wall).  chi_00 = 0 is the uniform mode."""
    out={}
    for m in range(mmax+1):
        z=list(jnp_zeros(m,qmax)) if qmax>0 else []
        out[m]=([0.0]+z) if m==0 else z
    return out

def overlap(m, j, chi, a):
    eps = 2*np.pi if m==0 else np.pi
    if abs(chi) < 1e-12:
        if m!=0: return 0.0
        return eps*a*a*(-jvp(0,j))/j
    d = j*j - chi*chi
    if abs(d) < 1e-9: d = 1e-9
    return eps*a*a*(-j*jvp(m,j)*jv(m,chi))/d

def lam_norm(m, chi, s, a, L):
    ang = 2*np.pi if m==0 else np.pi
    ax  = L if s==0 else L/2
    if abs(chi) < 1e-12:
        rad = a*a/2 if m==0 else 0.0
        if m==0: rad = a*a/2
    else:
        rad = (a*a/2)*jv(m,chi)**2*(1 - m*m/(chi*chi))
    return ang*rad*ax

def couple_block(m, Bh, Rh, a, L, qmax=4, smax=2, fcut=9000.0):
    """Return (freqs, vectors, head-index maps) for one angular order."""
    ib=np.where(Bh['m']==m)[0]; ir=np.where(Rh['m']==m)[0] if Rh is not None else np.array([],int)
    if len(ib)==0: return None
    chis=chi_table(m,qmax)[m]
    cav=[]
    for q,chi in enumerate(chis):
        for s in range(smax+1):
            w=C_A*np.sqrt((chi/a)**2+(s*np.pi/L)**2)
            if w/(2*np.pi) > fcut: continue
            Lm=lam_norm(m,chi,s,a,L)
            if Lm<=0: continue
            cav.append((chi,s,w,Lm))
    nb,nr,nc=len(ib),len(ir),len(cav)
    N=nb+nr; M=np.zeros(N); om2=np.zeros(N)
    for t,i in enumerate(ib): M[t]=Bh['mass'][i]; om2[t]=Bh['om'][i]**2
    for t,i in enumerate(ir): M[nb+t]=Rh['mass'][i]; om2[nb+t]=Rh['om'][i]**2
    C=np.zeros((N,nc))
    for jc,(chi,s,w,Lm) in enumerate(cav):
        for t,i in enumerate(ib):  C[t,jc]=overlap(m,Bh['j'][i],chi,a)
        for t,i in enumerate(ir):  C[nb+t,jc]=overlap(m,Rh['j'][i],chi,a)*((-1)**s)
    Lv=np.array([c[3] for c in cav]); wc2=np.array([c[2]**2 for c in cav])
    n=N+nc
    A=np.zeros((n,n)); Bm=np.zeros((n,n))
    A[:N,:N]=np.diag(M); Bm[:N,:N]=np.diag(M*om2); Bm[:N,N:]=-C
    A[N:,:N]=(RHO_A*C_A**2/Lv)[:,None]*C.T
    A[N:,N:]=np.eye(nc); Bm[N:,N:]=np.diag(wc2)
    w2,V=eig(Bm,A)
    ok=np.isfinite(w2)&(w2.real>1.0)&(np.abs(w2.imag)<1e-3*np.abs(w2.real))
    w2=w2[ok].real; V=V[:,ok].real
    o=np.argsort(w2)
    return dict(f=np.sqrt(w2[o])/(2*np.pi), V=V[:,o], ib=ib, ir=ir, nb=nb, nr=nr,
                nc=nc, cav=cav)
