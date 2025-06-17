import numpy as np

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

class Args:
    path_temp="./ai-temp"
    path_results = "./ai-results"
    path_data_save="./ai-data"
    path_data_base="./great-dataset"

    parameter_class = KITTIParameters