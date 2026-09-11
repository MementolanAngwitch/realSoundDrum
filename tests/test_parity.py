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

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden_js.json')

# The JS mode table is rounded to 4 dp to keep the page small, so frequencies
# carry ~1e-5 relative error before anything else happens.  These tolerances are
# set well inside what that permits, and far inside audibility.
RTOL_FREQ  = 2e-4        # mode frequencies
RTOL_SCALAR = 5e-3       # gmax, bend, dent, drive
ONSET_DB   = -55.0       # residual floor on the first 50 ms of audio
ONSET_S    = 0.05
BAND_DB    = 0.2         # third-octave agreement over the whole render


def _cases():
    with open(GOLDEN) as fh:
        return json.load(fh)


def _render(name):
    d = build(Params(**PRESETS[name]))
    y, r = render(d, impulse=PRESETS[name]['P'] * 1e-3)
    return d, y, r


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
    """The audio itself, over the first 50 ms.

    Compared here rather than over the whole render because the JS mode table is
    rounded to 4 dp to keep the page small, which leaves ~1e-5 relative error in
    every frequency.  Over 1.45 s that is up to a quarter cycle of accumulated
    phase at 16 kHz -- a large sample-by-sample residual that says nothing about
    whether the physics agrees.  Over 50 ms it is a few degrees, so this window
    is a real test of the waveform.
    """
    for name, g in _cases().items():
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
    """Third-octave magnitudes over the whole render.

    This is the parity statement that survives the phase drift above: if the two
    engines put the same energy in the same places for a second and a half, they
    are the same engine.
    """
    fs = 44100 / 7
    for name, g in _cases().items():
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


if __name__ == '__main__':
    for fn in (test_mode_frequencies, test_derived_quantities,
               test_strike_response, test_waveform_onset, test_spectrum):
        fn()
        print("ok  %s" % fn.__name__)
    # a readable summary too
    print()
    print("%-13s %6s %9s %8s %8s   onset vs JS" % ("case", "modes", "f1", "gmax", "bend"))
    for name, g in _cases().items():
        d, y, r = _render(name)
        js = np.array(g['y']); py = y[::7][:len(js)]
        k = int(ONSET_S * 44100 / 7)
        db = 20 * math.log10(np.sqrt(((py[:k] - js[:k]) ** 2).mean())
                             / np.sqrt((js[:k] ** 2).mean()))
        print("%-13s %6d %9.2f %8.4f %8.1f   %7.1f dB"
              % (name, d.n, d.f1, r['gmax'], r['bend'], db))
