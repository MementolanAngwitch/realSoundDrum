#!/usr/bin/env python3
"""Regenerate tests/golden_js.json from the browser engine.  Needs node.

The parity test compares the Python engine against numbers taken from the
JavaScript one.  Storing them means the test runs anywhere; this script is how
you refresh them after changing the JS.
"""
import os, re, json, subprocess, tempfile, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, 'web', 'physicalDrumBeta.html')
OUT  = os.path.join(ROOT, 'tests', 'golden_js.json')

# name -> (preset, overrides).  The unwired snare holds the cavity physics to
# the strict comparisons; the wired one is compared statistically, because a
# rattle is a nonlinear, partly chaotic process and the two engines start
# 1e-5 apart (the JS mode table is rounded to keep the page small).
CASES = {
    'floor tom':        ('floor tom', {}),
    'timpano':          ('timpano', {}),
    'steel sheet':      ('steel sheet', {}),
    'snare, no wires':  ('snare', {'wN': 0}),
    'latex sheet':      ('latex sheet', {}),
    'snare':            ('snare', {}),
}
WINDOWS = [(0.0, 0.02), (0.02, 0.06), (0.06, 0.15)]
BANDS = [(100, 500), (500, 2000), (2000, 6000), (6000, 16000)]

HARNESS = r"""
const cases = %s, WINDOWS = %s, BANDS = %s;
function bandsDb(y) {
  // energy per band per window, by direct DFT on a coarse grid -- slow but
  // dependency-free, and the same arithmetic the Python side does with an FFT
  const out = [];
  for (const [a, b] of WINDOWS) {
    const i0 = Math.floor(a * SR), i1 = Math.floor(b * SR), n = i1 - i0, row = [];
    for (const [lo, hi] of BANDS) {
      let e = 0;
      const k0 = Math.ceil(lo * n / SR), k1 = Math.floor(hi * n / SR);
      for (let k = k0; k <= k1; k++) {
        let re = 0, im = 0; const w = 2 * Math.PI * k / n;
        for (let t = 0; t < n; t++) { const v = y[i0 + t]; re += v * Math.cos(w * t); im -= v * Math.sin(w * t); }
        e += re * re + im * im;
      }
      row.push(10 * Math.log10(e + 1e-30));
    }
    out.push(row);
  }
  return out;
}
const out = {};
for (const name in cases) {
  const [preset, over] = cases[name];
  const pr = PRESETS.find(q => q.n === preset);
  const p = Object.assign({}, KIT[2], pr.p, over);
  const d = buildDrum(p);
  const r = renderHit(d, p.P * 1e-3, 1.0);
  out[name] = {
    n: d.n, nU: d.nU, nC: d.nC, f1: d.f1, tau: d.tau, fTop: d.fTop,
    stiff: d.stiff, airfrac: d.airfrac, wEff: d.wEff, beta: d.beta, tauL: d.tauL,
    fC: Array.from(d.fC.slice(0, 8)),
    om: Array.from(d.om.slice(0, 12)),
    sig: Array.from(d.sig.slice(0, 12)),
    gmax: r.gmax, bend: r.bend, dent: r.dent, drive: r.drive,
    tFrozen: r.tFrozen, bands: bandsDb(r.y),
    y: Array.from(r.y.filter((_, i) => i %% 7 === 0)).slice(0, 1200)
  };
}
console.log(JSON.stringify(out));
"""


def js_source():
    """Pull the engine out of the HTML: everything inside the one <script>."""
    html = open(HTML, encoding='utf-8').read()
    body = html.split('<script>', 1)[1].rsplit('</script>', 1)[0]
    # drop anything that touches the DOM -- we only want the physics
    cut = body.index('/* ------------------------------------------------------------------- audio */')
    return body[:cut]


if __name__ == '__main__':
    src = js_source() + HARNESS % (json.dumps(CASES), json.dumps(WINDOWS), json.dumps(BANDS))
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as fh:
        fh.write(src)
        path = fh.name
    try:
        res = subprocess.run(['node', path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    if res.returncode != 0:
        sys.exit("node failed:\n" + res.stderr[:2000])
    json.dump(json.loads(res.stdout), open(OUT, 'w'), indent=1)
    print("wrote %s (%d cases)" % (os.path.relpath(OUT, ROOT), len(CASES)))
