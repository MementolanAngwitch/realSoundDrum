"""
Snare wires: strands pressed against the far sheet, free to leave it.

Everywhere else in the engine a strike sets initial velocities and the modes run
free -- linear, closed form.  Wires are not that.  A strand rests on the
outside of the far sheet under a preload.  When the sheet pulls away faster than
the preload can make the strand follow, the strand LOSES CONTACT, flies, and
comes back and hits it.  The buzz is that train of collisions, and a collision
is a force that exists only while the two surfaces would otherwise overlap: a
genuine per-sample nonlinearity.

GEOMETRY.  Strands run parallel to the y axis, perpendicular to the strike axis
(theta = 0), at a fixed pitch, centred on the sheet.  That orientation is not
cosmetic.  The mode set is cos(m theta) only, which is exact for anything
symmetric about the strike axis -- and strands laid across that axis, each
centred on it, keep the whole configuration symmetric.  Laid along it instead,
the strands at +y and -y would see identical motion and hit in lockstep pairs.

EACH STRAND is a string of tension T and linear density mu, discretised into
WIRE_PTS point masses, anchored at both ends beyond the rim.  What holds it
against the sheet is the SNARE BED: the bearing edge is cut lower where the
strands cross it, so the anchors sit a depth `bed` past the sheet's plane.  Over
the flat middle of the sheet the strand is straight and tangent to it, and a
straight string exerts no normal force -- so the whole preload is carried where
the strand bends over the beds, next to the rim:

    F_end = T bed / l_s          at the first and last point of each strand
    F     = 0                    everywhere between

An earlier version gave every point a uniform load (a strand sagging onto a
bulging sheet).  That set one throw-off threshold, 85 g, everywhere along the
strand, and a sheet that moves 0.15 mm under the strands clears 85 g for about
20 ms -- a click, not a buzz.  With the beds, the middle of each strand rests on
the sheet with nothing but its neighbours' tension to bring it back, so it
leaves at any outward acceleration and returns at the string's own rate.  The
buzz lasts as long as the sheet moves, which is what a snare does.

THE CONTACT is a constraint, not a spring.  Each sample, every point either
touches with a pushing force and no overlap, or is clear with no force:

    eta = y_free - w_free                  overlap if nothing pushed back
    F >= 0,   eta - A F <= 0,   F . (eta - A F) = 0

A is the one-step compliance between every pair of points -- the sheet's, from
its modes, plus each strand point's own dt^2/m.  A push at one point moves the
sheet under all the others within the same sample, so the forces are found
together, as this linear complementarity problem (`solve_contact`).  A is
symmetric positive definite, so the answer is unique and both engines land on
it.  There is no contact stiffness to choose: a coil of 0.3 mm steel wound at
2.5 mm is ~9e8 N/m per metre radially, a ~4 us contact, far below one sample,
so at audio rate the strand is rigid and the bounce comes from the sheet's own
elasticity, which the modes already carry.

What was tried and why it is not here -- each looked fine until measured:

  * each point solved alone (the diagonal of A): twenty strands each pushed as
    if alone, 2.4x too hard in total, all let go the next sample, and chattered
    every sample with no strike at all;
  * the row sum of A (Gershgorin): stable, but every contact 3.1x too soft, and
    softer as points get closer together -- exactly backwards;
  * Newton restitution on top (a coefficient e at impact): NOT PASSIVE.  With
    many simultaneous contacts, Newton's law can create energy, and here it
    did: with every loss switched on, the sheet settled onto a plateau of
    constant energy and rattled forever.  It also produced a +34 dB high-band
    tail that looked exactly like a snare and was the energy leak.  The test
    that catches it is tests/test_physics.py::test_wires_are_passive.

EVERY STRAND IS ITS OWN.  An earlier version merged strands closer together
than the far sheet's modes can resolve into bundles, on the argument that the
sheet cannot tell them apart.  That argument is wrong, and measurably so: the
sheet cannot tell them apart, but the strands still decorrelate from each other,
and a bundle forces its strands to hit in lockstep -- a heavy lockstep bundle
clamps the sheet where separate light strands bounce.  The high-band excess
60-150 ms after the strike, which is the snare's signature, went

    4 bundles  +4 dB     8 bundles  +21 dB     10 bundles  +32 dB     20 strands  +35 dB

and did not converge until the bundling was gone.  Unbundled, it is converged
in the discretisation: +29 to +35 dB from 5 to 8 points per strand and a
10 to 14 kHz mode ceiling.

There is no contact stiffness to choose.  A Mylar sheet absorbs a 1.6 g/m strand's
momentum in ~27 us -- about one sample -- so what sets the collision is the
sheet's modal compliance, which the engine already has.  (A stiff penalty spring
would need the 5x oversampling the FA2023 paper used, for no audible gain.)

Units, because the one real bug in the first version of this was a unit: the
modes take an IMPULSE.  A contact force is newtons; its impulse over a sample is
F dt.  Without the dt every collision is crushed by 1/dt = 44100.
"""
import math

import numpy as np

from .constants import WIRE_PTS, WIRE_PITCH


WIRE_SPAN = 0.85          # fraction of each chord the strand covers


def geometry(a, n_strands, pts=WIRE_PTS, pitch=WIRE_PITCH, span=WIRE_SPAN):
    """Contact points for n_strands strands laid across the strike axis.

    Returns (r_frac, theta, ls, half, u) with r_frac, theta, u of shape
    (n_strands, pts), and the point spacing ls and half-length half, each of
    shape (n_strands,).
    """
    n = int(n_strands)
    xs = (np.arange(n) - (n - 1) / 2) * pitch
    xs = np.clip(xs, -0.95 * a, 0.95 * a)
    half = span * np.sqrt(np.maximum(a * a - xs * xs, 0.0))
    ls = 2 * half / (pts + 1)
    u = np.arange(1, pts + 1)[None, :] * ls[:, None] - half[:, None]    # y
    x = np.repeat(xs[:, None], pts, axis=1)
    r = np.sqrt(x * x + u * u) / a
    th = np.arctan2(u, x)
    return r, th, ls, half, u


def rest_shape(u, half, bed):
    """Where the strand would lie with no sheet: straight, `bed` past the sheet's
    plane, between anchors that are also `bed` past it."""
    return np.full(u.shape, float(bed))


def elastic(y, y0, T, ls):
    """Tension force on each point, from its neighbours, relative to the rest shape.

    Ends fixed at the anchors, y = y0 there.  Positive = into the sheet.
    """
    d = y - y0
    lap = -2 * d
    lap[:, 1:] += d[:, :-1]
    lap[:, :-1] += d[:, 1:]
    # the fixed ends contribute (0 - y0_end) where y0 at the ends is zero
    return (T / ls)[:, None] * lap


def elastic_matrix(T, ls, pts):
    """E(y) = E(0) + Lw @ y.ravel(): the strand tension force is linear in y."""
    nb = len(ls)
    n = nb * pts
    Lw = np.zeros((n, n))
    for b in range(nb):
        k = T[b] / ls[b] if np.ndim(T) else T / ls[b]
        o = b * pts
        for i in range(pts):
            Lw[o + i, o + i] = -2 * k
            if i > 0:
                Lw[o + i, o + i - 1] = k
            if i < pts - 1:
                Lw[o + i, o + i + 1] = k
    return Lw


def solve_contact(A, eta, active=None, max_iter=200):
    """The exact contact forces for one sample.

    Find F >= 0 with  g = eta - A F <= 0  and  F . g = 0:  every point either
    touches with a pushing force and no overlap, or is clear with no force.  A
    is the one-step compliance (sheet + strand), symmetric positive definite, so
    the solution exists and is unique -- which is also why the browser, solving
    the same problem by projected Gauss-Seidel, lands on the same numbers.

    Active-set (principal pivoting), warm-started from the previous sample's
    contacts; one or two solves per sample in practice.
    """
    n = len(eta)
    act = (eta > 0) if active is None else active.copy()
    F = np.zeros(n)
    for it in range(max_iter):
        F[:] = 0.0
        a = np.flatnonzero(act)
        if len(a):
            F[a] = np.linalg.solve(A[np.ix_(a, a)], eta[a])
        neg = F < -1e-15
        if neg.any():
            # drop one at a time (the most negative) once it starts to thrash
            if it < 20:
                act &= ~neg
            else:
                act[np.argmin(F)] = False
            continue
        g = eta - A @ F
        viol = (~act) & (g > 1e-15 * max(1.0, float(np.abs(eta).max())))
        if viol.any():
            if it < 20:
                act |= viol
            else:
                act[np.flatnonzero(viol)[np.argmax(g[viol])]] = True
            continue
        return np.maximum(F, 0.0), act
    raise RuntimeError("contact solve did not converge")
