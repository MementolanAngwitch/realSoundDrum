"""
The Python engine and the browser engine must agree.

They are two independent implementations of the same physics, which is worth a
great deal — a disagreement is a bug in one of them, and that is how several
real ones were found.  But an oracle only works if something checks it: these
two drifted badly once, with the browser gaining bending stiffness, materials,
the in-plane tension lag and the (q,v) integrator while the Python kept none of
them, and nothing anywhere failed.  This is the thing that fails.

Reference values live in golden_js.json so the test needs no node.  Refresh them
with `python tools/extract_js_golden.py` after changing the JavaScript.

    pytest tests/            (or just: python tests/test_parity.py)
"""
import os
import sys
import json
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physicaldrum import Params, PRESETS, build, render          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, 'tests', 'golden_js.json')

# The JS mode table is rounded to 4 dp to keep the page small, so frequencies
# carry ~1e-5 relative error before anything else happens.  These tolerances are
# set well inside what that permits, and far inside audibility.
RTOL_FREQ  = 2e-4        # mode frequencies
RTOL_SCALAR = 5e-3       # gmax, bend, dent, drive
ONSET_DB   = -55.0       # residual floor on the first 50 ms of audio
ONSET_S    = 0.05
BAND_DB    = 0.2         # band agreement, linear cases
BAND_WIRED_DB = 1.0      # band agreement for a rattle (measured 0.21 dB)


def _cases():
    with open(GOLDEN) as fh:
        return json.load(fh)


def _params(name):
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    from extract_js_golden import CASES
    preset, over = CASES[name]
    return Params(**{**PRESETS[preset], **over})


_CACHE = {}


def _render(name):
    if name not in _CACHE:
        p = _params(name)
        d = build(p)
        y, r = render(d, impulse=p.P * 1e-3)
        _CACHE[name] = (d, y, r)
    return _CACHE[name]


def _wired(g):
    return g['tFrozen'] < 1.4


def bands_db(y):
    """Energy per band per window -- the same numbers the harness computes."""
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    from extract_js_golden import WINDOWS, BANDS
    out = []
    for a, b in WINDOWS:
        s = y[int(a * 44100): int(b * 44100)]
        n = len(s)
        X = np.abs(np.fft.rfft(s)) ** 2
        k = np.arange(len(X))
        row = []
        for lo, hi in BANDS:
            sel = (k >= math.ceil(lo * n / 44100)) & (k <= math.floor(hi * n / 44100))
            row.append(10 * math.log10(X[sel].sum() + 1e-30))
        out.append(row)
    return np.array(out)


def test_mode_frequencies():
    for name, g in _cases().items():
        d, _, _ = _render(name)
        assert d.n == g['n'], "%s: %d modes vs %d in JS" % (name, d.n, g['n'])
        py = d.om[:12]
        js = np.array(g['om'])
        rel = np.abs(py - js) / np.maximum(np.abs(js), 1e-12)
        assert rel.max() < RTOL_FREQ, \
            "%s: worst mode frequency differs by %.2e" % (name, rel.max())


def test_derived_quantities():
    for name, g in _cases().items():
        d, _, _ = _render(name)
        for key, got in (('f1', d.f1), ('tau', d.tau), ('fTop', d.info['f_top']),
                         ('stiff', d.info['stiffness']),
                         ('airfrac', d.info['air_fraction']),
                         ('wEff', d.info['tip_eff']),
                         ('beta', d.beta), ('tauL', d.tauL)):
            want = g[key]
            rel = abs(got - want) / max(abs(want), 1e-12)
            assert rel < RTOL_SCALAR, \
                "%s: %s = %.6g, JS says %.6g (%.2e)" % (name, key, got, want, rel)


def test_strike_response():
    for name, g in _cases().items():
        _, _, r = _render(name)
        for key, got in (('gmax', r['gmax']), ('dent', r['dent']),
                         ('drive', r['drive'])):
            want = g[key]
            rel = abs(got - want) / max(abs(want), 1e-12)
            assert rel < RTOL_SCALAR, \
                "%s: %s = %.6g, JS says %.6g" % (name, key, got, want)
        # bend is a difference of logs; compare in cents, not relatively
        assert abs(r['bend'] - g['bend']) < 2.0, \
            "%s: bend %.1f cents vs %.1f" % (name, r['bend'], g['bend'])


def test_waveform_onset():
    """The audio itself, over the first 50 ms -- for the linear cases.

    Over 50 ms the 4 dp rounding of the JS mode table is a few degrees of phase,
    so this is a real test of the waveform.  Not for a rattle: that is a
    nonlinear, partly chaotic process, and two engines starting 1e-5 apart part
    company within milliseconds, exactly as two real snares would.  The wired
    case is held to test_band_energies and test_strands_settle_together.
    """
    for name, g in _cases().items():
        if _wired(g):
            continue
        _, y, _ = _render(name)
        js = np.array(g['y'])
        py = y[::7][:len(js)]
        k = int(ONSET_S * 44100 / 7)
        err = np.sqrt(((py[:k] - js[:k]) ** 2).mean())
        ref = np.sqrt((js[:k] ** 2).mean())
        db = 20 * math.log10(err / ref) if ref > 0 else -999
        assert db < ONSET_DB, \
            "%s: onset residual %.1f dB, floor is %.1f dB" % (name, db, ONSET_DB)


def test_spectrum():
    """Third-octave magnitudes over the first 190 ms, decimated -- linear cases."""
    fs = 44100 / 7
    for name, g in _cases().items():
        if _wired(g):
            continue
        _, y, _ = _render(name)
        js = np.array(g['y'])
        py = y[::7][:len(js)]
        win = np.hanning(len(js))
        Y1 = np.abs(np.fft.rfft(py * win))
        Y2 = np.abs(np.fft.rfft(js * win))
        fr = np.fft.rfftfreq(len(js), 1 / fs)
        lo = 60.0
        while lo * 1.26 < fs / 2:
            m = (fr >= lo) & (fr < lo * 1.26)
            if m.sum() >= 3:
                e1 = np.sqrt((Y1[m] ** 2).sum())
                e2 = np.sqrt((Y2[m] ** 2).sum())
                if e2 > 0:
                    d = abs(20 * math.log10(e1 / e2))
                    assert d < BAND_DB, \
                        "%s: %.0f Hz band differs by %.2f dB" % (name, lo, d)
            lo *= 1.26


def test_band_energies():
    """Energy in four bands, in three windows, at the full sample rate.

    For the linear cases this is tight.  For the rattle it is the parity
    statement that survives chaos: two engines whose strands hit at different
    instants must still put the same energy in the same places.
    """
    for name, g in _cases().items():
        _, y, _ = _render(name)
        d = np.abs(bands_db(y) - np.array(g['bands'])).max()
        tol = BAND_WIRED_DB if _wired(g) else BAND_DB
        assert d < tol, "%s: band energy differs by %.2f dB" % (name, d)


def test_strands_settle_together():
    """Both engines must decide the rattle is over at the same moment."""
    for name, g in _cases().items():
        _, _, r = _render(name)
        got = r.get('t_frozen', 1.45)
        assert abs(got - g['tFrozen']) <= 0.0101, \
            "%s: strands settle at %.3f s here, %.3f s in JS" % (name, got, g['tFrozen'])


if __name__ == '__main__':
    for fn in (test_mode_frequencies, test_derived_quantities, test_strike_response,
               test_waveform_onset, test_spectrum, test_band_energies,
               test_strands_settle_together):
        fn()
        print("ok  %s" % fn.__name__)
    # a readable summary too
    print()
    print("%-16s %6s %9s %8s %8s %11s %10s %8s" % (
        "case", "modes", "f1", "gmax", "bend", "onset", "bands", "settle"))
    for name, g in _cases().items():
        d, y, r = _render(name)
        js = np.array(g['y']); py = y[::7][:len(js)]
        k = int(ONSET_S * 44100 / 7)
        db = 20 * math.log10(np.sqrt(((py[:k] - js[:k]) ** 2).mean())
                             / np.sqrt((js[:k] ** 2).mean()))
        bd = np.abs(bands_db(y) - np.array(g['bands'])).max()
        onset = "(rattle)" if _wired(g) else "%.1f dB" % db
        print("%-16s %6d %9.2f %8.4f %8.1f %11s %7.2f dB %6.0f ms" % (
            name, d.n, d.f1, r['gmax'], r['bend'], onset, bd, 1000 * r.get('t_frozen', 1.45)))
