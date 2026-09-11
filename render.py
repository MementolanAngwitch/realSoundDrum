#!/usr/bin/env python3
"""
Render presets to .wav.

    python render.py                       every preset, one file each
    python render.py "floor tom" snare     just those
    python render.py --all --out sounds/   somewhere else

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


def one(name, outdir):
    p = PRESETS[name]
    d = build(Params(**p))
    y, r = render(d, impulse=p['P'] * 1e-3)
    path = os.path.join(outdir, name.replace(' ', '_') + '.wav')
    wav.write(path, SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    return d, r, path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*', help='preset names (default: all)')
    ap.add_argument('--out', default=HERE, help='output directory')
    ap.add_argument('--list', action='store_true', help='list presets and exit')
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

    print("%-13s %-10s %6s %8s %7s %7s %7s" %
          ("preset", "material", "modes", "f1", "top", "stiff", "bend"))
    for n in names:
        d, r, path = one(n, a.out)
        print("%-13s %-10s %6d %7.1f %6.1fk %6.0f%% %6.0f  ->  %s"
              % (n, d.info['material'], d.n, d.f1, d.info['f_top'] / 1000,
                 100 * d.info['stiffness'], r['bend'], os.path.basename(path)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
