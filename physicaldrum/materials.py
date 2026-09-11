"""Sheet materials.

Representative values, not measurements of any particular sheet.  `eta` is the
loss factor stored x1000 so the UI slider is an integer; `sheet.py` divides.

    rho  kg/m^3   density
    E    Pa       Young's modulus -- sets bending rigidity AND how hard the
                  sheet resists being stretched (the tension-modulation beta)
    nu   -        Poisson ratio
    eta  x1e-3    loss factor; damping is sigma = sigma_0 + eta omega/2
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    rho: float
    E: float
    nu: float
    eta: float          # x1e-3


MATERIALS = [
    Material('Mylar',     1390, 4.0e9,  0.40, 15),
    Material('Kevlar',    1440, 30e9,   0.35, 12),
    Material('Calfskin',  1000, 0.6e9,  0.40, 30),
    Material('Latex',      950, 0.02e9, 0.48, 80),
    Material('Paper',      800, 3.0e9,  0.30, 40),
    Material('Silk',      1300, 8.0e9,  0.35, 25),
    Material('Aluminium', 2700, 69e9,   0.33,  2),
    Material('Steel',     7850, 200e9,  0.30,  1),
]
BY_NAME = {m.name.lower(): i for i, m in enumerate(MATERIALS)}


def material(which):
    """Accept an index, a name, or a Material."""
    if isinstance(which, Material):
        return which
    if isinstance(which, str):
        return MATERIALS[BY_NAME[which.lower()]]
    return MATERIALS[int(which)]
