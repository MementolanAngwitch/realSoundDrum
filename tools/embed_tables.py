#!/usr/bin/env python3
"""Write the generated tables into the browser page.

The page carries its data inline so it works from a double-click with no server.
Two tables come from Python:

    AIR   the air added-mass surface A(mode, r), from air_table.npz
    CHI   zeros of J_m' -- the rigid-wall radial wavenumbers of the cavity

Run after `python build_tables.py` whenever either changes.  The mode table
(MODES) is left alone: it is not regenerated here.

    python tools/embed_tables.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from physicaldrum import tables                                     # noqa: E402
from physicaldrum.cavity import neumann_zeros                       # noqa: E402
from physicaldrum.constants import J_CAV_MAX, CHI_PAD               # noqa: E402

HTML = os.path.join(ROOT, 'web', 'physicalDrumBeta.html')

# The cavity needs zeros of J_m' past the highest coupled sheet mode.  m = 0
# couples every m = 0 mode of the sheet, up to the top of the mode table; every
# other order couples only j <= J_CAV_MAX.
M_MAX = 64
CHI0_MAX = tables.J_MAX + CHI_PAD + 5
CHI_MAX = J_CAV_MAX + CHI_PAD + 5


def air_js():
    rs, A = tables.air_grid()
    dr = rs[1] - rs[0]
    body = ",".join("%.4g" % x for x in A.T.ravel())      # mode-major
    return ("const AIR={r0:%.4f,dr:%.6f,nr:%d,\nA:[%s]};" % (rs[0], dr, len(rs), body))


def chi_js():
    rows = []
    for m in range(M_MAX + 1):
        z = neumann_zeros(m, CHI0_MAX if m == 0 else CHI_MAX)
        rows.append("[" + ",".join("%.8g" % x for x in z) + "]")
    return "const CHI=[\n" + ",\n".join(rows) + "];"


def main():
    html = open(HTML, encoding='utf-8').read()
    lines = html.split('\n')
    a0 = next(i for i, l in enumerate(lines) if l.startswith('const AIR={'))
    a1 = next(i for i in range(a0, len(lines)) if lines[i].endswith(']};'))
    new = [air_js()]
    # an existing CHI block directly after AIR is replaced; otherwise inserted
    rest = a1 + 1
    if lines[rest].startswith('const CHI=['):
        rest = next(i for i in range(rest, len(lines)) if lines[i].endswith(']];'))
        rest += 1
    new.append(chi_js())
    lines[a0:rest] = new
    out = '\n'.join(lines)
    open(HTML, 'w', encoding='utf-8').write(out)
    print("AIR and CHI written into %s (%d bytes)" % (os.path.relpath(HTML, ROOT), len(out)))


if __name__ == '__main__':
    main()
