import numpy as np
from typing import Optional
from singular_data import SingularData
from pyproj import Transformer
from scipy.linalg import cho_factor, cho_solve # type: ignore

def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of a vector."""
    return np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]]
    )

def euler_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body -> world rotation, ZYX convention (yaw-pitch-roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array(
        [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp, cp * sr, cp * cr]]
    )

def wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

# P[0:3, 0:3] *= 0.5 # on position
# P[3:6, 3:6] *= 0.1 ** 2 # on velocity
# P[6:9, 6:9] *= (1.0 * np.pi / 180) ** 2 # on attitude
# P[9:12, 9:12] *= 0.1 ** 2 # on accel-bias
# P[12:15, 12:15] *= (0.01 * np.pi / 180) ** 2 # on gyro-bias

class KalmanFilter:
    """
    15-state loosely-coupled inertial KF
    state = [p(3) v(3) euler(3) b_a(3) b_g(3)]^T
    """

    def __init__(self, dt: float, first: SingularData, P: np.ndarray) -> None:
        self.dt = dt
        self.g = np.array([0.0, 0.0, 9.80665]) # gravity (ENU, +Up)

        self.x = np.zeros(15)
        self.x[3:6] = np.array([first.ve, first.vn, first.vu]) # initial ENU velocity, in reality would be 0
        self.x[6:9] = np.array([first.roll, first.pitch, first.yaw]) # radians
        # biases start at 0

        # covariance
        self.P = P

        # process-noise (continuous)
        self.q_acc   = 0.05 ** 2 # (m/s^2)^2
        self.q_gyro  = (0.02 * np.pi / 180) ** 2 # (rad / s)^2
        self.q_b_acc = 5e-6 # accel-bias random-walk
        self.q_b_gyro = 5e-8 # gyro-bias random-walk

        # measurement noise (orientation only now)
        self.R_meas = np.diag([
            (0.5 * np.pi / 180) ** 2, # roll
            (0.5 * np.pi / 180) ** 2, # pitch
            (1.0 * np.pi / 180) ** 2  # yaw
        ])
        # shape 3x3

        # store first GPS for "ground truth" plotting
        self.first_gps = np.array([first.lat, first.lon, first.alt])
        zone = int((first.lon + 180.) / 6) + 1
        epsg = f"EPSG:{32600 + zone}" if first.lat >= 0 else f"EPSG:{32700 + zone}"
        self.tf = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
        self.utm0 = np.array(self.tf.transform(first.lon, first.lat))

    # ---------------------------------------------------------------------------
    # PUBLIC STEP
    # ---------------------------------------------------------------------------

    def step(self, imu: SingularData, update_measurement: bool = True) -> None:
        """Predict with raw IMU; optionally correct with roll/pitch/yaw measurement."""
        self._predict(imu)
        if update_measurement:
            self._update_measurement(np.array([imu.roll, imu.pitch, imu.yaw]))

    # ---------------------------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------------------------

    def _predict(self, d: SingularData) -> None:
        dt = self.dt

        # state shortcuts
        p = self.x[0:3]
        v = self.x[3:6]
        phi = self.x[6:9]
        b_a = self.x[9:12]
        b_g = self.x[12:15]

        # measured specific force & angular-rate (body)
        f_b = np.array([d.ax, d.ay, d.az]) - b_a
        w_b = np.array([d.wx, d.wy, d.wz]) - b_g

        # attitude & rotation
        R = euler_to_R(*phi)
        f_n = R @ f_b - self.g # world linear acceleration

        # state propagation
        p_new = p + v * dt + 0.5 * f_n * dt ** 2
        v_new = v + f_n * dt
        phi_new = phi + w_b * dt
        phi_new = wrap_pi(phi_new)

        self.x[0:3] = p_new
        self.x[3:6] = v_new
        self.x[6:9] = phi_new
        # biases stay the same

        # build continuous-time Jacobian F_c
        Fc = np.zeros((15, 15))
        Fc[0:3, 3:6] = np.eye(3)
        Fc[3:6, 6:9] = -R @ skew(f_b) # dv/dφ
        Fc[3:6, 9:12] = -R # dv/db_a
        Fc[6:9, 12:15] = -np.eye(3) # dφ/db_g

        # discretise: Phi = I + Fc*dt (1st order)
        Phi = np.eye(15) + Fc * dt

        # discrete process-noise Qd
        Qd = np.zeros((15, 15))
        Qd[3:6, 3:6] = np.eye(3) * self.q_acc * dt
        Qd[6:9, 6:9] = np.eye(3) * self.q_gyro * dt
        Qd[9:12, 9:12] = np.eye(3) * self.q_b_acc * dt
        Qd[12:15, 12:15] = np.eye(3) * self.q_b_gyro * dt

        # position noise - integrate twice
        Qd[0:3, 0:3] = np.eye(3) * self.q_acc * dt ** 3 / 3
        Qd[0:3, 3:6] = np.eye(3) * self.q_acc * dt ** 2 / 2
        Qd[3:6, 0:3] = Qd[0:3, 3:6]

        # covariance propagation P = Phi P Phi^T + Qd
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = self._pd(self.P)

    # ---------------------------------------------------------------------------
    # MEASUREMENT  (orientation only, 3x1)
    # ---------------------------------------------------------------------------

    def _update_measurement(self, z_rpy: np.ndarray) -> None:
        # Measurement model: H maps state to measurements
        # We measure roll, pitch, yaw + assume vz =~ 0 for ground vehicle
        H = np.zeros((4, 15))
        H[0:3, 6:9] = np.eye(3) # measure the three Euler angles
        H[3, 5] = 1.0 # measure vz (vertical velocity)

        # Expected measurements
        h = np.zeros(4)
        h[0:3] = self.x[6:9] # current attitude estimate
        h[3] = self.x[5] # current vz estimate

        # Innovation (measurement residual)
        z_full = np.append(z_rpy, 0.0) # append vz = 0 measurement
        y = np.zeros(4)
        y[0:3] = wrap_pi(z_rpy - h[0:3]) # wrap angle differences
        y[3] = z_full[3] - h[3] # vz difference

        # Measurement noise covariance (add vz noise)
        R_full = np.zeros((4, 4))
        R_full[0:3, 0:3] = self.R_meas
        R_full[3, 3] = (0.1) ** 2 # vz noise: (0.1 m/s)^2

        S = H @ self.P @ H.T + R_full
        K = self._gain(S, H)

        # State update
        self.x += K @ y
        self.x[6:9] = wrap_pi(self.x[6:9]) # wrap angles after update

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_full @ K.T
        self.P = self._pd(self.P)

    # ---------------------------------------------------------------------------
    # NUMERICAL HELPERS
    # ---------------------------------------------------------------------------

    @staticmethod
    def _pd(M: np.ndarray) -> np.ndarray:
        """Force matrix to be symmetric positive-definite (very small bump if needed)."""
        M = 0.5 * (M + M.T)
        eig, V = np.linalg.eigh(M)
        tiny = 1e-10
        eig[eig < tiny] = tiny
        return V @ np.diag(eig) @ V.T

    def _gain(self, S: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Return Kalman gain P H^T S^-1 with a fallback."""
        if cho_factor is not None:
            c, low = cho_factor(S)
            K = self.P @ H.T
            K = cho_solve((c, low), K.T).T # solves S X = P H^T
        else: # fall back, use pseudo-inverse
            K = self.P @ H.T @ np.linalg.pinv(S)
        return K

    # ---------------------------------------------------------------------------
    # SMALL UTILITY GETTERS
    # ---------------------------------------------------------------------------

    def position(self) -> np.ndarray:
        return self.x[0:3].copy()

    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    def attitude(self) -> np.ndarray:
        return self.x[6:9].copy()
    
    def orientation(self) -> np.ndarray:
        return self.x[6:9].copy()

    # ---------------------------------------------------------------------------
    # GPS ground-truth helper (unchanged)
    # ---------------------------------------------------------------------------

    def gps_displacement(self, d: SingularData) -> np.ndarray:
        cur = np.array(self.tf.transform(d.lon, d.lat))
        dx, dy = cur - self.utm0
        dz = d.alt - self.first_gps[2]
        return np.array([dx, dy, dz])