# ---------------------------------------------------------------------------
# generate_circle.py - produces perfect KITTI-like IMU data
# ---------------------------------------------------------------------------
# The folder structure created is
#
#   sim_circle/
#       timestamps.txt
#       dataformat.txt          (purely informational)
#       data/
#           0000000000.txt
#           0000000001.txt
#           ...
#
# Each *.txt contains the 30 floating-point fields expected by your
# SingularData.read_data().
# ---------------------------------------------------------------------------

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# ---------------------------------------------------------------------------
# user-tunable parameters
# ---------------------------------------------------------------------------

FOLDER          = Path("sim_circle")   # output directory
RATE_HZ         = 100                  # sample frequency
DT              = 1.0 / RATE_HZ
RADIUS          = 50.0                 # metres (circle radius)
TURN_RATE       = 0.10                 # rad/s  (angular speed around circle)
LAT0, LON0, ALT = 37.0, -122.0, 10.0   # start position (degrees, metres)
G               = 9.80665              # gravity  (m/s², +Up in sensor frame)

# ---------------------------------------------------------------------------
# derived constants
# ---------------------------------------------------------------------------

T_CIRCLE   = 2.0 * np.pi / TURN_RATE       # seconds for one full lap
N_SAMPLES  = int(round(T_CIRCLE * RATE_HZ))
V          = RADIUS * TURN_RATE            # constant ground speed
ACC_CENT   = RADIUS * TURN_RATE ** 2       # centripetal accel magnitude

# Earth radius for small-angle lat/lon conversion (WGS-84 mean)
R_EARTH = 6_378_137.0

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def enu_to_ll(dx: float, dy: float) -> tuple[float, float]:
    """
    Convert small ENU offsets (dx east, dy north) in metres
    to (lat, lon) degrees relative to (LAT0, LON0).
    """
    d_lat = dy / R_EARTH
    d_lon = dx / (R_EARTH * np.cos(np.radians(LAT0)))
    return LAT0 + np.degrees(d_lat), LON0 + np.degrees(d_lon)


def unit_circle_sim() -> list[list[str]]:
    """
    Build one full revolution; return a list of 30-field strings
    (already converted to str) – one per sample.
    """
    rows: list[list[str]] = []

    for k in range(N_SAMPLES):
        t = k * DT
        theta = TURN_RATE * t                     # central angle (0 … 2π)

        # -------- world (ENU) position & velocity ----------------------------
        x = RADIUS * np.cos(theta)               # east
        y = RADIUS * np.sin(theta)               # north
        z = 0.0

        ve = -RADIUS * TURN_RATE * np.sin(theta)  # east velocity
        vn =  RADIUS * TURN_RATE * np.cos(theta)  # north velocity
        vu = 0.0

        # -------- orientation -------------------------------------------------
        # heading tangent to path
        yaw   = np.arctan2(vn, ve)               # rad
        pitch = 0.0
        roll  = 0.0

        # -------- body-frame acceleration (specific force) -------------------
        # forward  x-axis  : 0  (constant speed)
        # left     y-axis  : +a_c
        # up       z-axis  : +g
        ax = 0.0
        ay = ACC_CENT
        az = G

        # -------- body-frame angular rates -----------------------------------
        wx = 0.0
        wy = 0.0
        wz = TURN_RATE                           # yaw rate (around +Up)

        # same values in alternative axes (KITTI stores wf, wl, wu)
        wf, wl, wu = wx, wy, wz

        # -------- lat / lon from ENU -----------------------------------------
        lat, lon = enu_to_ll(x, y)

        # -------- pack the 30 fields -----------------------------------------
        row = [
            f"{lat:.10f}",
            f"{lon:.10f}",
            f"{ALT:.4f}",
            f"{roll:.10f}",
            f"{pitch:.10f}",
            f"{yaw:.10f}",
            f"{vn:.10f}",
            f"{ve:.10f}",
            f"{V:.10f}",          # forward speed
            f"0.0",               # vl  (leftward speed)
            f"{vu:.10f}",
            f"{ax:.10f}",
            f"{ay:.10f}",
            f"{az:.10f}",
            f"0.0",               # af  (forward accel)
            f"{ay:.10f}",         # al  (left accel)
            f"0.0",               # au  (up accel)
            f"{wx:.10f}",
            f"{wy:.10f}",
            f"{wz:.10f}",
            f"{wf:.10f}",
            f"{wl:.10f}",
            f"{wu:.10f}",
            "0.0", "0.0",         # pos_accuracy, vel_accuracy
            "0", "0", "0", "0", "0"    # navstat, numsats, posmode, velmode, orimode
        ]
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# write files
# ---------------------------------------------------------------------------

def main() -> None:
    FOLDER.mkdir(exist_ok=True)
    (FOLDER / "data").mkdir(exist_ok=True)

    rows = unit_circle_sim()

    # ---------- timestamps.txt ----------------------------------------------
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    with (FOLDER / "timestamps.txt").open("w") as f_ts:
        for k in range(N_SAMPLES):
            f_ts.write((t0 + timedelta(seconds=k * DT)).isoformat() + "\n")

    # ---------- dataformat.txt  (informational) -----------------------------
    fmt = (
        "lat lon alt roll pitch yaw "
        "vn ve vf vl vu "
        "ax ay az af al au "
        "wx wy wz wf wl wu "
        "posAcc velAcc navstat numsats posmode velmode orimode"
    )
    (FOLDER / "dataformat.txt").write_text(fmt + "\n")

    # ---------- individual sample files ------------------------------------
    for k, row in enumerate(rows):
        fname = FOLDER / "data" / f"{k:010d}.txt"
        fname.write_text(" ".join(row) + "\n")


if __name__ == "__main__":
    main()