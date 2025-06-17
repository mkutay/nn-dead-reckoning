# generate_rectangle.py
# ---------------------------------------------------------------------------
# Generates a perfect KITTI-style IMU/GPS sequence of a vehicle tracing
# a rectangle: straight → in-place 90° turn → straight → … (one loop).
# No sensor noise, ideal specific-force and angular-rate readings.
# ---------------------------------------------------------------------------

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# ----------------------------------------------------------------------------
# USER PARAMETERS
# ----------------------------------------------------------------------------

FOLDER     = Path("sim_rectangle")  # output directory
DATA_DIR   = FOLDER / "data"
RATE_HZ    = 100                    # IMU & GPS sample rate
DT         = 1.0 / RATE_HZ
LENGTH_X   = 100.0                  # meters, rectangle long side
LENGTH_Y   =  50.0                  # meters, rectangle short side
V_FWD      =   5.0                  # m/s, constant forward speed on straights
TURN_ANGLE =  np.pi/2               # radians, 90° at each corner
W_TURN     =  np.pi/2               # rad/s, in-place yaw rate → 1 s per 90°
ALT        =  10.0                  # meters, constant altitude
LAT0, LON0 = 37.0, -122.0            # deg, center of ENU frame
G          =  9.80665               # m/s², +Up gravity

# ----------------------------------------------------------------------------
# DERIVED QUANTITIES
# ----------------------------------------------------------------------------

# number of samples on each straight
N_x = int(round((LENGTH_X / V_FWD) * RATE_HZ))
N_y = int(round((LENGTH_Y / V_FWD) * RATE_HZ))
# number of samples for each 90° in-place turn
T_turn = TURN_ANGLE / W_TURN        # seconds (here = 1.0 s)
N_t   = int(round(T_turn * RATE_HZ))

# sides: 0→east, 1→north, 2→west, 3→south
lengths = [LENGTH_X, LENGTH_Y, LENGTH_X, LENGTH_Y]
counts  = [N_x,       N_y,       N_x,       N_y]
headings = [0.0, np.pi/2, np.pi, -np.pi/2]  # yaw on each straight

# Total samples
TOTAL = sum(counts) + 4 * N_t

# Earth radius for small‐angle LL↔ENU
R_EARTH = 6_378_137.0

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------

def enu_to_ll(dx: float, dy: float) -> tuple[float, float]:
    """
    Convert small ENU offsets (dx east, dy north) in meters
    to (lat, lon) degrees relative to (LAT0, LON0).
    """
    d_lat = dy / R_EARTH
    d_lon = dx / (R_EARTH * np.cos(np.radians(LAT0)))
    return LAT0 + np.degrees(d_lat), LON0 + np.degrees(d_lon)


def make_rectangle_rows() -> list[list[str]]:
    """
    Return a list of rows (each is list of 30 strings) for the rectangle path.
    """
    rows: list[list[str]] = []
    # current ENU pos & yaw
    x, y, z = 0.0, 0.0, 0.0
    yaw = 0.0

    for side in range(4):
        # --- STRAIGHT SEGMENT ---
        hdg = headings[side]
        N_straight = counts[side]
        ve_const = V_FWD * np.cos(hdg)
        vn_const = V_FWD * np.sin(hdg)

        for i in range(N_straight):
            # advance position
            x += ve_const * DT
            y += vn_const * DT

            # lat/lon
            lat, lon = enu_to_ll(x, y)

            # IMU / GPS fields
            roll, pitch = 0.0, 0.0
            yaw = hdg
            vn, ve, vf = vn_const, ve_const, V_FWD
            vl, vu = 0.0, 0.0

            # body-frame specific force (no linear accel, only gravity in Z)
            ax, ay, az = 0.0, 0.0, G
            af, al, au = ax, ay, 0.0

            # angular rates (no rotation)
            wx, wy, wz = 0.0, 0.0, 0.0
            wf, wl, wu = wx, wy, wz

            row = [
                f"{lat:.10f}", f"{lon:.10f}", f"{ALT:.4f}",
                f"{roll:.10f}", f"{pitch:.10f}", f"{yaw:.10f}",
                f"{vn:.10f}", f"{ve:.10f}", f"{vf:.10f}",
                f"{vl:.10f}", f"{vu:.10f}",
                f"{ax:.10f}", f"{ay:.10f}", f"{az:.10f}",
                f"{af:.10f}", f"{al:.10f}", f"{au:.10f}",
                f"{wx:.10f}", f"{wy:.10f}", f"{wz:.10f}",
                f"{wf:.10f}", f"{wl:.10f}", f"{wu:.10f}",
                "0.0", "0.0",  # pos_accuracy, vel_accuracy
                "0", "0", "0", "0", "0"  # navstat, numsats, posmode, velmode, orimode
            ]
            rows.append(row)

        # --- IN-PLACE 90° TURN ---
        yaw_start = yaw
        for i in range(N_t):
            # increment yaw
            yaw = yaw_start + W_TURN * DT * (i+1)
            # wrap to (−π, π]
            yaw = (yaw + np.pi) % (2*np.pi) - np.pi

            # position unchanged
            lat, lon = enu_to_ll(x, y)

            roll, pitch = 0.0, 0.0
            vn = ve = vf = vl = vu = 0.0

            # no linear accel, only gravity
            ax, ay, az = 0.0, 0.0, G
            af = al = au = 0.0

            # pure yaw‐rate
            wx, wy, wz = 0.0, 0.0, W_TURN
            wf, wl, wu = wx, wy, wz

            row = [
                f"{lat:.10f}", f"{lon:.10f}", f"{ALT:.4f}",
                f"{roll:.10f}", f"{pitch:.10f}", f"{yaw:.10f}",
                f"{vn:.10f}", f"{ve:.10f}", f"{vf:.10f}",
                f"{vl:.10f}", f"{vu:.10f}",
                f"{ax:.10f}", f"{ay:.10f}", f"{az:.10f}",
                f"{af:.10f}", f"{al:.10f}", f"{au:.10f}",
                f"{wx:.10f}", f"{wy:.10f}", f"{wz:.10f}",
                f"{wf:.10f}", f"{wl:.10f}", f"{wu:.10f}",
                "0.0", "0.0",
                "0", "0", "0", "0", "0"
            ]
            rows.append(row)

        # wrap yaw for next straight
        yaw = (yaw_start + TURN_ANGLE + np.pi) % (2*np.pi) - np.pi

    return rows

# ----------------------------------------------------------------------------
# WRITE OUT THE DATASET
# ----------------------------------------------------------------------------

def main():
    FOLDER.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    rows = make_rectangle_rows()
    assert len(rows) == TOTAL, f"Expected {TOTAL} rows, got {len(rows)}"

    # timestamps.txt
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    with open(FOLDER / "timestamps.txt", "w") as f:
        for i in range(TOTAL):
            ts = t0 + timedelta(seconds=i * DT)
            f.write(ts.isoformat() + "\n")

    # dataformat.txt (informational)
    fmt = (
        "lat lon alt roll pitch yaw "
        "vn ve vf vl vu "
        "ax ay az af al au "
        "wx wy wz wf wl wu "
        "posAcc velAcc navstat numsats posmode velmode orimode"
    )
    (FOLDER / "dataformat.txt").write_text(fmt + "\n")

    # individual sample files
    for i, row in enumerate(rows):
        fn = DATA_DIR / f"{i:010d}.txt"
        fn.write_text(" ".join(row) + "\n")

    print(f"Generated {TOTAL} perfect rectangle samples in '{FOLDER}/'")

if __name__ == "__main__":
    main()