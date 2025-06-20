import numpy as np

from beyond_dataset import BeyondDataset # type: ignore
from midair_dataset import MidAirDataset # type: ignore

class Parameters:
    g = np.array([0, 0, -9.80665])
    """gravity vector"""

    P_dim = 21
    """covariance dimension"""

    Q_dim = 18
    """process noise covariance dimension"""

    # Process noise covariance
    cov_omega = 1e-3
    """gyro covariance"""
    cov_acc = 1e-2
    """accelerometer covariance"""
    cov_b_omega = 6e-9
    """gyro bias covariance"""
    cov_b_acc = 2e-4
    """accelerometer bias covariance"""
    cov_Rot_c_i = 1e-9
    """car to IMU orientation covariance"""
    cov_t_c_i = 1e-9
    """car to IMU translation covariance"""

    cov_lat = 0.2
    """Zero lateral velocity covariance"""
    cov_up = 300
    """Zero lateral velocity covariance"""

    cov_Rot0 = 1e-3
    """initial pitch and roll covariance"""
    cov_b_omega0 = 6e-3
    """initial gyro bias covariance"""
    cov_b_acc0 = 4e-3
    """initial accelerometer bias covariance"""
    cov_v0 = 1e-1
    """initial velocity covariance"""
    cov_Rot_c_i0 = 1e-6
    """initial car to IMU pitch and roll covariance"""
    cov_t_c_i0 = 5e-3
    """initial car to IMU translation covariance"""

    # numerical parameters
    n_normalize_rot = 100
    """timestamp before normalizing orientation"""
    n_normalize_rot_c_i = 1000
    """timestamp before normalizing car to IMU orientation"""

    def __init__(self, **kwargs):
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class KITTIParameters(Parameters):
    # gravity vector
    g = np.array([0, 0, -9.80655])

    cov_omega = 2e-4
    cov_acc = 1e-3
    cov_b_omega = 1e-8
    cov_b_acc = 1e-6
    cov_Rot_c_i = 1e-8
    cov_t_c_i = 1e-8
    cov_Rot0 = 1e-6
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-8
    cov_b_acc0 = 1e-3
    cov_Rot_c_i0 = 1e-5
    cov_t_c_i0 = 1e-2
    cov_lat = 1
    cov_up = 10

    def __init__(self, **kwargs):
        super(KITTIParameters, self).__init__(**kwargs)
        self.set_param_attr()

    def set_param_attr(self):
        attr_list = [a for a in dir(KITTIParameters) if not a.startswith('__') and not callable(getattr(KITTIParameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(KITTIParameters, attr))

class MidAirParameters(Parameters):
    """
    Parameters optimized for MidAir drone dataset.
    Drones have different motion characteristics compared to ground vehicles:
    - Higher angular velocities and accelerations
    - More 3D motion (including vertical movement)
    - Different sensor noise characteristics
    - More aggressive maneuvers
    
    Using more conservative values for training stability.
    """
    # gravity vector (same as standard)
    g = np.array([0, 0, -9.80665])

    # Process noise covariance - conservative values for training stability
    cov_omega = 2e-3  # Moderate gyro covariance for drone maneuvers
    cov_acc = 5e-3    # Moderate accelerometer covariance for dynamic flight
    cov_b_omega = 5e-9  # Conservative gyro bias covariance
    cov_b_acc = 1e-4    # Conservative accelerometer bias covariance  
    cov_Rot_c_i = 5e-9  # Conservative IMU to body orientation covariance
    cov_t_c_i = 5e-9    # Conservative IMU to body translation covariance

    # Velocity constraints - moderate values for drones
    cov_lat = 1.0      # Moderate lateral velocity covariance
    cov_up = 10        # Moderate vertical velocity covariance

    # Initial covariances - conservative uncertainty for stable initialization
    cov_Rot0 = 2e-3        # Moderate initial orientation uncertainty
    cov_b_omega0 = 5e-3    # Moderate initial gyro bias uncertainty  
    cov_b_acc0 = 2e-3      # Conservative initial accelerometer bias uncertainty
    cov_v0 = 2e-1          # Moderate initial velocity uncertainty
    cov_Rot_c_i0 = 5e-6    # Conservative initial IMU to body orientation uncertainty
    cov_t_c_i0 = 2e-3      # Conservative initial IMU to body translation uncertainty

    def __init__(self, **kwargs):
        super(MidAirParameters, self).__init__(**kwargs)
        self.set_param_attr()

    def set_param_attr(self):
        attr_list = [a for a in dir(MidAirParameters) if not a.startswith('__') and not callable(getattr(MidAirParameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(MidAirParameters, attr))

class Args:
    path_temp="./ai-temp"
    path_results = "./ai-results-midair"
    path_data_save="./ai-data-midair"
    path_data_base="./MidAir"

    parameter_class = MidAirParameters
    dataset_class = MidAirDataset

    epochs = 200
    seq_dim = 6000

    continue_training = True