"""Starting points, not identities.

Every number here is one you could have dialled in yourself.  Each is calibrated
to a ~4 mm peak dent -- a real stroke -- subject to staying inside the range
where the tension model is honest.  Impulse is NOT the thing to hold fixed when
you change the striker: S scales roughly as (a/w)^2, so re-anchor on the dent.

2026-09-23: the bass drum was re-anchored from 147.70 to 37.92 mN.s.  Nothing
about it changed -- the dent readout was wrong.  It used to leave out the m = 0
modes, which on a centre-struck drum are nearly all of the dent, so "3.25 mm"
had really been 14.45 mm, with the pitch bend pinned at its clamp.  The other
presets are struck off centre and moved from 4.00 mm to 4.1-4.4 mm; left alone.
"""
PRESETS = {
    'bass drum':   dict(mat=0, a=0.280, T=1526,  h=14.0, s0=2.4, eta=30, P=37.92,  w=22.0, hard=2.2, s=2,  d=400,  Tr=0.92, hr=12.0, port=0,  ang=72, fmax=11000),
    'rack tom':    dict(mat=0, a=0.203, T=3330,  h=10.0, s0=1.2, eta=16, P=5.84,   w=3.7,  hard=0,   s=8,  d=260,  Tr=0.85, hr=7.5,  port=0,  ang=60, fmax=16000),
    'floor tom':   dict(mat=0, a=0.164, T=3471,  h=10.0, s0=1.2, eta=16, P=5.83,   w=3.5,  hard=0,   s=8,  d=340,  Tr=0.85, hr=7.5,  port=0,  ang=60, fmax=16000),
    'snare':       dict(mat=0, a=0.178, T=5516,  h=7.5,  s0=3.0, eta=22, P=6.91,   w=4.3,  hard=0,   s=12, d=140,  Tr=1.55, hr=3.0,  port=10, ang=60, fmax=16000, wN=20),
    'timpano':     dict(mat=2, a=0.330, T=1917,  h=7.5,  s0=0.8, eta=12, P=34.59,  w=14.0, hard=1.4, s=6,  d=700,  Tr=0,    hr=7.5,  port=0,  ang=55, fmax=9000),
    'frame drum':  dict(mat=2, a=0.220, T=2600,  h=9.0,  s0=2.0, eta=30, P=17.05,  w=8.0,  hard=0.6, s=14, d=60,   Tr=0,    hr=7.5,  port=0,  ang=60, fmax=14000),
    'steel sheet': dict(mat=7, a=0.300, T=9000,  h=6.0,  s0=0.4, eta=1,  P=27.71,  w=4.0,  hard=0,   s=10, d=3000, Tr=0,    hr=6.0,  port=0,  ang=60, fmax=16000),
    'alu plate':   dict(mat=6, a=0.120, T=26000, h=10.0, s0=0.6, eta=2,  P=40.40,  w=3.0,  hard=0,   s=9,  d=3000, Tr=0,    hr=10.0, port=0,  ang=60, fmax=16000),
    'paper drum':  dict(mat=4, a=0.150, T=900,   h=5.0,  s0=3.0, eta=40, P=1.59,   w=4.0,  hard=0,   s=11, d=150,  Tr=0.9,  hr=5.0,  port=0,  ang=60, fmax=16000),
    'latex sheet': dict(mat=3, a=0.200, T=400,   h=20.0, s0=2.0, eta=80, P=5.40,   w=10.0, hard=0.5, s=9,  d=3000, Tr=0,    hr=20.0, port=0,  ang=60, fmax=9000),
    'silk hoop':   dict(mat=5, a=0.110, T=3000,  h=3.0,  s0=1.5, eta=25, P=3.66,   w=2.5,  hard=0,   s=13, d=3000, Tr=0,    hr=3.0,  port=0,  ang=60, fmax=16000),
    'kevlar disc': dict(mat=1, a=0.140, T=9000,  h=6.0,  s0=1.0, eta=12, P=11.76,  w=3.0,  hard=0,   s=8,  d=3000, Tr=0,    hr=6.0,  port=0,  ang=60, fmax=16000),
}
