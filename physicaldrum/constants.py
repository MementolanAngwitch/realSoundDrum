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
