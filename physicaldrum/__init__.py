"""physicalDrumBeta — a clamped circular sheet, solved from physical properties.

    from physicaldrum import Params, build, render, PRESETS
    d = build(Params(**PRESETS['snare']))
    y, report = render(d, impulse=PRESETS['snare']['P'] * 1e-3)
"""
from .materials import MATERIALS, material          # noqa: F401
from .engine import Params, Drum, build, render     # noqa: F401
from .presets import PRESETS                        # noqa: F401

__all__ = ['MATERIALS', 'material', 'Params', 'Drum', 'build', 'render', 'PRESETS']
