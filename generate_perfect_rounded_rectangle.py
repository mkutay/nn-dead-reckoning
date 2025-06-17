# generate_rounded_rectangle.py
# ---------------------------------------------------------------------------
# Generates a perfect KITTI-style IMU/GPS sequence of a vehicle tracing
# a rectangle with smooth 90° corner arcs of fixed radius.  No sensor noise.
# Folder structure created:
#   sim_rectangle_rounded/
#     timestamps.txt
#     dataformat.txt
#     data/
#       0000000000.txt
#       0000000001.txt
#       ...
# ---------------------------------------------------------------------------

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# ----------------------------------------------------------------------------
# USER PARAMETERS
# ----------------------------------------------------------------------------
FOLDER      = Path("sim_rectangle_rounded")
DATA_DIR    = FOLDER / "data"
RATE_HZ     = 100                   # sample rate
DT          = 1.0 / RATE_HZ
LENGTH_X    = 100.0                 # rectangle long side (m)
LENGTH_Y    =  50.0                 # rectangle short side (m)
V_FWD       =   5.0                 # constant forward speed (m/s)
R_C         =  10.0                 # corner radius (m), must be < min(LENGTH_X, LENGTH_Y)/2
ALT         =  10.0                 # constant altitude (m)
LAT0, LON0  = 37.0, -122.0          # reference lat/lon for ENU origin
G           =   9.80665             # gravity (m/s^2, +Up in body frame)

# ----------------------------------------------------------------------------
# DERIVED QUANTITIES
# ----------------------------------------------------------------------------
# lengths of trimmed straights (each end loses R_C)
Lx = max(LENGTH_X - 2*R_C, 0.0)
Ly = max(LENGTH_Y - 2*R_C, 0.0)

# number of samples on each straight
N_x = int(round((Lx / V_FWD) * RATE_HZ))
N_y = int(round((Ly / V_FWD) * RATE_HZ))

# number of samples on each 90° corner arc
T_arc = (R_C * (np.pi/2)) / V_FWD
N_arc = int(round(T_arc * RATE_HZ))

# total samples
TOTAL = 1 + 4*(N_x + N_arc + N_y + N_arc)  # +1 for initial sample

# earth radius for small-angle ENU -> LL
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

def make_rounded_rectangle() -> list[list[str]]:
    """
    Build one lap around a rectangle with rounded corners.
    Returns a list of 30-field rows (strings).
    """
    rows: list[list[str]] = []
    # starting ENU pos (we start at the downstream end of the last corner,
    # so that after four straights+arcs we close the loop)
    x, y = R_C, 0.0
    heading = 0.0  # initial yaw = east

    # initial sample at t=0
    def append_row(x, y, heading, ax, ay, az, wx, wy, wz):
        lat, lon = enu_to_ll(x, y)
        # forward & lateral speeds in ENU
        ve = V_FWD * np.cos(heading)
        vn = V_FWD * np.sin(heading)
        vf = V_FWD
        vl = 0.0
        vu = 0.0
        af = ax
        al = ay
        au = 0.0
        wf, wl, wu = wx, wy, wz
        row = [
            f"{lat:.10f}", f"{lon:.10f}", f"{ALT:.4f}",
            f"{0.0:.10f}", f"{0.0:.10f}", f"{heading:.10f}",
            f"{vn:.10f}", f"{ve:.10f}", f"{vf:.10f}",
            f"{vl:.10f}", f"{vu:.10f}",
            f"{ax:.10f}", f"{ay:.10f}", f"{az:.10f}",
            f"{af:.10f}", f"{al:.10f}", f"{au:.10f}",
            f"{wx:.10f}", f"{wy:.10f}", f"{wz:.10f}",
            f"{wf:.10f}", f"{wl:.10f}", f"{wu:.10f}",
            "0.0", "0.0", "0", "0", "0", "0", "0"
        ]
        rows.append(row)

    # 1) initial sample: no motion yet, measure gravity only
    append_row(x, y, heading, 0.0, 0.0, G, 0.0, 0.0, 0.0)

    # define the 4 straight+arc blocks in order: East, North, West, South
    for side in range(4):
        # determine straight length & heading
        if side % 2 == 0:
            Ls = Lx
            hdg = 0.0 if side==0 else np.pi  # 0=east, π=west
        else:
            Ls = Ly
            hdg = np.pi/2 if side==1 else -np.pi/2  # π/2=north, -π/2=south

        # 2) straight segment (trimmed by R_C at each end):
        vx = V_FWD * np.cos(hdg)
        vy = V_FWD * np.sin(hdg)
        for i in range(1, N_x+1 if side%2==0 else N_y+1):
            # advance
            x += vx * DT
            y += vy * DT
            # no linear accel in body‐X/Y, only gravity in Z
            append_row(x, y, hdg, 0.0, 0.0, G, 0.0, 0.0, 0.0)

        # 3) quarter‐circle arc, turning left 90°
        # compute center of rotation
        # left‐unit = (-sin(hdg), +cos(hdg))
        ux, uy = -np.sin(hdg), np.cos(hdg)
        cx, cy = x + R_C * ux, y + R_C * uy

        # starting arc‐angle φ_start = atan2(y−cy, x−cx)
        phi_start = np.arctan2(y - cy, x - cx)
        # total turn = +π/2 (left turn)
        for k in range(1, N_arc+1):
            frac = (k / N_arc) * (np.pi / 2)
            phi = phi_start + frac
            # new pos
            x = cx + R_C * np.cos(phi)
            y = cy + R_C * np.sin(phi)
            # heading increments by same frac
            new_heading = hdg + frac
            # wrap into (−π, +π]
            new_heading = (new_heading + np.pi) % (2*np.pi) - np.pi
            # specific‐force in body: lateral centripetal = V²/R, plus gravity
            a_c = V_FWD**2 / R_C
            append_row(x, y, new_heading, 0.0, a_c, G, 0.0, 0.0, V_FWD / R_C)

        # update heading for next straight
        heading = (hdg + np.pi/2 + np.pi) % (2*np.pi) - np.pi

    return rows

# ----------------------------------------------------------------------------
# WRITE OUTPUT
# ----------------------------------------------------------------------------
def main():
    FOLDER.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    rows = make_rounded_rectangle()
    assert len(rows) <= 1_000_000, "too many samples"

    # timestamps.txt
    t0 = datetime(2025,1,1,12,0,0)
    with (FOLDER/"timestamps.txt").open("w") as f:
        for i in range(len(rows)):
            ts = t0 + timedelta(seconds=i * DT)
            f.write(ts.isoformat() + "\n")

    # dataformat.txt (informational only)
    fmt = (
        "lat lon alt roll pitch yaw "
        "vn ve vf vl vu "
        "ax ay az af al au "
        "wx wy wz wf wl wu "
        "posAcc velAcc navstat numsats posmode velmode orimode"
    )
    (FOLDER/"dataformat.txt").write_text(fmt + "\n")

    # per-sample data files
    for i, row in enumerate(rows):
        fn = DATA_DIR / f"{i:010d}.txt"
        fn.write_text(" ".join(row) + "\n")

    print(f"Generated {len(rows)} perfect rounded-rectangle samples in '{FOLDER}'")

if __name__ == "__main__":
    main()