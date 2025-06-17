# generate_rounded_rectangle_noisy.py

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# ----------------------------------------------------------------------------
# OUTPUT FOLDER
# ----------------------------------------------------------------------------
FOLDER    = Path("sim_rectangle_rounded_noisy")
DATA_DIR  = FOLDER / "data"

# ----------------------------------------------------------------------------
# SIMULATION PARAMETERS
# ----------------------------------------------------------------------------
RATE_HZ    = 150                   # samples per second
DT         = 1.0 / RATE_HZ
LENGTH_X   = 200.0                 # long side of rectangle, meters
LENGTH_Y   =  80.0                 # short side, meters
R_C        =  25.0                 # corner radius, meters
V_FWD      =   6.0                 # constant forward speed, m/s
ALT        =  10.0                 # constant altitude, m
LAT0, LON0 = 37.0, -122.0          # reference lat/lon for ENU origin
G          =   9.80665             # gravity, m/s² (+Up in body frame)

# ----------------------------------------------------------------------------
# NOISE PARAMETERS (change these to increase/decrease noise)
# ----------------------------------------------------------------------------
SEED           = 2
rng            = np.random.default_rng(SEED)
ACC_NOISE_STD  = 0.02   # accelerometer noise, m/s² (std. dev.)
GYRO_NOISE_STD = 0.01   # gyroscope noise,   rad/s (std. dev.)
ORI_NOISE_STD  = 0.002  # orientation noise, rad (std. dev.)

# ----------------------------------------------------------------------------
# DERIVED QUANTITIES
# ----------------------------------------------------------------------------
# length of trimmed straights = side length minus 2×corner‐radius
Lx = max(LENGTH_X - 2*R_C, 0.0)
Ly = max(LENGTH_Y - 2*R_C, 0.0)

# sample counts on each straight and each quarter‐circle arc
N_x   = int(round((Lx / V_FWD) * RATE_HZ))
N_y   = int(round((Ly / V_FWD) * RATE_HZ))
T_arc = (R_C * (np.pi/2)) / V_FWD
N_arc = int(round(T_arc * RATE_HZ))

# ENU→LL conversion constant
R_EARTH = 6_378_137.0  # mean Earth radius, meters

# headings for the 4 sides: east, north, west, south
HEADINGS      = [0.0, np.pi/2, np.pi, -np.pi/2]
COUNTS_STRAIT = [N_x, N_y, N_x, N_y]

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def enu_to_ll(dx: float, dy: float) -> tuple[float, float]:
    """
    Convert small ENU offsets (dx east, dy north) [m]
    back to (lat, lon) degrees around (LAT0, LON0).
    """
    d_lat = dy / R_EARTH
    d_lon = dx / (R_EARTH * np.cos(np.radians(LAT0)))
    return LAT0 + np.degrees(d_lat), LON0 + np.degrees(d_lon)

def wrap_to_pi(angle: float) -> float:
    """Wrap a single angle to (-π, +π]."""
    return (angle + np.pi) % (2*np.pi) - np.pi

def make_rows() -> list[list[str]]:
    """
    Build all samples (each a list of 30 string-fields) for
    one lap around a rectangle with rounded corners.
    """
    rows: list[list[str]] = []

    # start at (x=R_C, y=0), heading east
    x, y = R_C, 0.0

    # 1) initial sample at rest: orientation=0, no motion, just gravity
    def append_row(lat, lon, heading, ax, ay, az, wx, wy, wz):
        """Helper to append one row (with noise injected)."""
        # perfect measurement + noise
        roll_m  = rng.normal(0.0, ORI_NOISE_STD)
        pitch_m = rng.normal(0.0, ORI_NOISE_STD)
        yaw_m   = heading + rng.normal(0.0, ORI_NOISE_STD)

        ax_m = ax + rng.normal(0.0, ACC_NOISE_STD)
        ay_m = ay + rng.normal(0.0, ACC_NOISE_STD)
        az_m = az + rng.normal(0.0, ACC_NOISE_STD)

        wx_m = wx + rng.normal(0.0, GYRO_NOISE_STD)
        wy_m = wy + rng.normal(0.0, GYRO_NOISE_STD)
        wz_m = wz + rng.normal(0.0, GYRO_NOISE_STD)

        # GPS-style velocities (perfect, no noise here)
        ve = V_FWD * np.cos(heading)
        vn = V_FWD * np.sin(heading)
        vf = V_FWD
        vl = 0.0
        vu = 0.0

        wf, wl, wu = wx_m, wy_m, wz_m  # same in KITTI alt axes

        row = [
            f"{lat:.10f}", f"{lon:.10f}", f"{ALT:.4f}",
            f"{roll_m:.10f}", f"{pitch_m:.10f}", f"{yaw_m:.10f}",
            f"{vn:.10f}", f"{ve:.10f}", f"{vf:.10f}",
            f"{vl:.10f}", f"{vu:.10f}",
            f"{ax_m:.10f}", f"{ay_m:.10f}", f"{az_m:.10f}",
            f"{ax_m:.10f}", f"{ay_m:.10f}", f"{0.0:.10f}",  # af, al, au
            f"{wx_m:.10f}", f"{wy_m:.10f}", f"{wz_m:.10f}",
            f"{wf:.10f}", f"{wl:.10f}", f"{wu:.10f}",
            "0.0", "0.0",  # pos_accuracy, vel_accuracy
            "0", "0", "0", "0", "0"  # navstat, numsats, posmode, velmode, orimode
        ]
        rows.append(row)

    # initial
    lat0, lon0 = enu_to_ll(x, y)
    append_row(lat0, lon0, 0.0, 0.0, 0.0, G, 0.0, 0.0, 0.0)

    # loop over 4 sides
    for side in range(4):
        hdg = HEADINGS[side]
        N_st = COUNTS_STRAIT[side]

        # 2) straight segment (trim by R_C at each end)
        vx = V_FWD * np.cos(hdg)
        vy = V_FWD * np.sin(hdg)
        for _ in range(N_st):
            x += vx * DT
            y += vy * DT
            lat, lon = enu_to_ll(x, y)
            # on a straight: no body-frame accel except gravity
            append_row(lat, lon, hdg, 0.0, 0.0, G, 0.0, 0.0, 0.0)

        # 3) quarter-circle arc (turn left by +90°)
        # center of rotation is R_C to the *left* of the heading
        ux, uy = -np.sin(hdg), np.cos(hdg)
        cx, cy = x + R_C * ux, y + R_C * uy
        phi0 = np.arctan2(y - cy, x - cx)
        a_c = V_FWD**2 / R_C
        w_z = V_FWD / R_C

        for k in range(1, N_arc+1):
            frac = (k / N_arc) * (np.pi/2)
            phi  = phi0 + frac
            x    = cx + R_C * np.cos(phi)
            y    = cy + R_C * np.sin(phi)
            new_hdg = wrap_to_pi(hdg + frac)
            lat, lon = enu_to_ll(x, y)
            # body-frame: centripetal accel on y-axis + gravity on z
            append_row(lat, lon, new_hdg,
                       0.0, a_c, G,
                       0.0, 0.0, w_z)

    return rows

# ----------------------------------------------------------------------------
# WRITE OUT THE FOLDER
# ----------------------------------------------------------------------------
def main():
    FOLDER.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    rows = make_rows()
    N = len(rows)

    # timestamps.txt
    t0 = datetime(2025,1,1,12,0,0)
    with (FOLDER/"timestamps.txt").open("w") as fts:
        for i in range(N):
            ts = t0 + timedelta(seconds=i*DT)
            fts.write(ts.isoformat() + "\n")

    # dataformat.txt (informational only)
    fmt = (
        "lat lon alt roll pitch yaw "
        "vn ve vf vl vu "
        "ax ay az af al au "
        "wx wy wz wf wl wu "
        "posAcc velAcc navstat numsats posmode velmode orimode"
    )
    (FOLDER/"dataformat.txt").write_text(fmt + "\n")

    # per-sample files
    for i, row in enumerate(rows):
        fn = DATA_DIR / f"{i:010d}.txt"
        fn.write_text(" ".join(row) + "\n")

    print(f"Generated {N} noisy rounded-rectangle samples in '{FOLDER}'")

if __name__ == "__main__":
    main()