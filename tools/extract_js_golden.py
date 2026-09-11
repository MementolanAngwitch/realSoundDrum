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

CASES = ['floor tom', 'timpano', 'steel sheet', 'snare', 'latex sheet']

HARNESS = r"""
const cases = %s;
const out = {};
for (const name of cases) {
  const pr = PRESETS.find(q => q.n === name);
  const p = Object.assign({}, KIT[2], pr.p);
  const d = buildDrum(p);
  const r = renderHit(d, p.P * 1e-3, 1.0);
  out[name] = {
    n: d.n, nU: d.nU, nC: d.nC, f1: d.f1, tau: d.tau, fTop: d.fTop,
    stiff: d.stiff, airfrac: d.airfrac, wEff: d.wEff, beta: d.beta, tauL: d.tauL,
    fC: Array.from(d.fC.slice(0, 8)),
    om: Array.from(d.om.slice(0, 12)),
    sig: Array.from(d.sig.slice(0, 12)),
    gmax: r.gmax, bend: r.bend, dent: r.dent, drive: r.drive,
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
    src = js_source() + HARNESS % json.dumps(CASES)
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
