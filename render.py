#!/usr/bin/env python3
"""
Render presets to .wav.

    python render.py                       every preset, one file each
    python render.py "floor tom" snare     just those
    python render.py snare --ab            also the same drum with the wires off
    python render.py --out sounds/         somewhere else

Nothing here is a drum by definition; the preset names are places to start.
"""
import os
import sys
import argparse

import numpy as np
import scipy.io.wavfile as wav

from physicaldrum import Params, PRESETS, build, render
from physicaldrum.constants import SR

HERE = os.path.dirname(os.path.abspath(__file__))


def one(name, outdir, **over):
    p = {**PRESETS[name], **over}
    d = build(Params(**p))
    y, r = render(d, impulse=p['P'] * 1e-3)
    tag = name.replace(' ', '_') + ('_no_wires' if over.get('wN') == 0 else '')
    path = os.path.join(outdir, tag + '.wav')
    wav.write(path, SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    return d, r, path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*', help='preset names (default: all)')
    ap.add_argument('--out', default=HERE, help='output directory')
    ap.add_argument('--list', action='store_true', help='list presets and exit')
    ap.add_argument('--ab', action='store_true',
                    help='for presets with wires, also render them with the wires off')
    a = ap.parse_args(argv)

    if a.list:
        for n in PRESETS:
            print(n)
        return 0

    names = a.names or list(PRESETS)
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        sys.exit("unknown preset(s): %s\nknown: %s"
                 % (', '.join(unknown), ', '.join(PRESETS)))
    os.makedirs(a.out, exist_ok=True)

    print("%-13s %-10s %6s %8s %7s %7s %7s %7s" %
          ("preset", "material", "modes", "f1", "top", "stiff", "bend", "wires"))
    jobs = [(n, {}) for n in names]
    if a.ab:
        jobs += [(n, {'wN': 0}) for n in names if PRESETS[n].get('wN', 0) > 0]
    for n, over in jobs:
        d, r, path = one(n, a.out, **over)
        w = "%d" % d.wire['n'] if d.wire else "-"
        print("%-13s %-10s %6d %7.1f %6.1fk %6.0f%% %6.0f %7s  ->  %s"
              % (n, d.info['material'], d.n, d.f1, d.info['f_top'] / 1000,
                 100 * d.info['stiffness'], r['bend'], w, os.path.basename(path)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
