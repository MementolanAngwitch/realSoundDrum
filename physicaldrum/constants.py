"""Physical and engine constants, in one place so the JS and Python agree."""
import math

RHO_AIR = 1.204          # kg/m^3
C_AIR   = 343.0          # m/s
K_AIR   = RHO_AIR * C_AIR * C_AIR      # gamma*P0, the bulk modulus of air

SR   = 44100
DUR  = 1.45              # seconds rendered per strike

MAXM    = 1500           # hard cap on modes per sheet
KEEP_DB = 40             # acoustic prune threshold, dB below the loudest mode
J_KEEP  = 55             # keep every mode below this j regardless (they carry S)
DEAD_DB = 80             # a mode is dropped once it is this far down

RAD_C     = 3.0e-5       # radiation-damping coefficient
MIC_RESO  = 0.30         # how much of the far sheet reaches a near-side mic
MIC_PORT  = 0.75         # ditto for the vent
PORT_LEN  = 1.7          # vent end-correction factor, L_eff = PORT_LEN * r

PICKUP_FR = 12                  # index into the radial grid for a contact pickup
PICKUP_TH = math.pi / 2

# Tension modulation multiplies each mode's frequency by gamma.  With the band
# reaching 16 kHz a gamma of 1.38 would push the top modes past Nyquist, where
# the recurrence folds them to arbitrary frequencies.  Modes that could ever
# exceed it are simply not modulated -- they sit above 13.8 kHz, where a few
# hundred cents of glide is inaudible.  GAMMA_MAX is separately an honesty
# limit: past ~800 cents small-strain theory is no longer telling the truth.
NYQ       = 0.995 * math.pi * SR
GAMMA_MAX = 1.6
OM_MOD    = NYQ / GAMMA_MAX

# Output stage -- the recording chain, not the sheet.  A close mic sees crest
# factors of 50-100; peak-normalising leaves the body 30 dB down.
RMS_TARGET = 0.16
RMS_WIN    = 0.25
DRIVE_MAX  = 8.0

# The cavity (cavity.py).  Cavity modes below F_DYN are real degrees of freedom
# that can resonate with the sheets; above it they are folded in as added mass,
# summed over every axial order in closed form.  Sheet modes with j <= J_CAV are
# coupled through the cavity in full; above that a sheet pattern decays within
# a fraction of the depth, cannot reach the far sheet, and sees the cavity as a
# half-space -- so it gets the same air layer as the open side, diagonally.
# J_CAV is held to at least 4a/L so that coth(alpha L) is 1 to 0.1% for
# everything left out.  CHI_PAD: how far past the top coupled j the radial sum
# runs; each term falls as chi^-5.  Q_CAV: wall and viscothermal loss on the
# cavity's own modes -- a chosen value, not measured.
F_DYN     = 5000.0
# ...but no higher than F_DYN_RATIO times the fastest sheet mode the cavity
# couples.  Folding a cavity mode into added mass errs as (omega_sheet /
# omega_cavity)^2, so what matters is the ratio, not the absolute frequency: a
# 70 cm kettle has ~200 cavity modes below 5 kHz, and all but a handful sit far
# above anything its slow sheet does.  Measured against F_DYN = 20 kHz: 0.74
# cents worst on the ten lowest sheet modes of any preset, and the timpano
# builds 10x faster.
F_DYN_RATIO = 3.0
J_CAV_MIN = 20.0
J_CAV_MAX = 60.0
CHI_PAD   = 30.0
Q_CAV     = 50.0
S0_CAV    = 1.0

# Snare wires (wires.py).  Strand geometry and loss are chosen, not measured.
# The strand defaults -- 1.6 g/m, 30 N -- are the values used by the modal snare
# model of the Forum Acusticum 2023 paper "Simulation of the snare-membrane
# collision in modal form"; nothing here is fitted to a recording.
WIRE_PTS   = 6           # contact points per strand (converged: see wires.py)
WIRE_PITCH = 3.5e-3      # m between neighbouring strands
WIRE_SIG   = 40.0        # 1/s, strand loss
F_WIRE     = 14000.0     # far-sheet modes the wires can excite, up to this
WIRE_QUIET = 60.0        # dB below peak (in acceleration) at which the strands are left
                         # at rest; -58.6 dB residual against never freezing
