#!/usr/bin/env python3
"""Regenerate the two precomputed tables.  See physicaldrum/tables.py."""
import time, json, os, numpy as np
from physicaldrum import tables

if __name__ == '__main__':
    t0 = time.time()
    M, J = tables.build_modes()
    print("modes : %4d modes, j up to %.1f  ->  %s  (%.1fs)"
          % (len(J), J[-1], os.path.basename(tables.MODES_PATH), time.time() - t0))
    t1 = time.time()
    A = tables.build_air(M, J)
    print("air   : %s over r in [%.2f, %.2f]  ->  %s  (%.0fs)"
          % (A.shape, tables.R_LO, tables.R_HI,
             os.path.basename(tables.AIR_PATH), time.time() - t1))
    d = json.load(open(tables.MODES_PATH))
    assert len(d['M']) == len(d['J']) == len(d['JP']) == A.shape[1]
    assert abs(d['J'][0] - 2.404825557695773) < 1e-9, "first zero should be j_01"
    assert d['JP'][0] < 0, "J_0'(j_01) is negative -- the sign was lost"
    assert np.all(np.diff(d['J']) >= 0), "table must be sorted by j"
    print("checks: ok")
