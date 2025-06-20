# type: ignore

from collections import namedtuple
from termcolor import cprint
import numpy as np
import torch
import os
import h5py
import glob

from base_dataset import BaseDataset
from utils import *

class MidAirDataset(BaseDataset):
    """
    MidAir Dataset loader for IMU and ground truth data.
    
    This class loads data from MidAir HDF5 files which contain:
    - IMU data: accelerometer and gyroscope measurements
    - Ground truth: position, attitude (quaternions), velocity, acceleration, angular_velocity
    - GPS data: position and velocity (optional)
    
    The data is processed and saved in the same format as KITTI dataset:
    - t: time vector from start (seconds)
    - u: IMU measurements [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
    - ang_gt: ground truth Euler angles [roll, pitch, yaw] (radians)
    - p_gt: ground truth position [x, y, z] (meters)
    - v_gt: ground truth velocity [vx, vy, vz] (m/s)
    """
    
    # Bundle into an easy-to-access structure
    GroundTruth = namedtuple('GroundTruth', ['position', 'attitude', 'velocity', 'acceleration', 'angular_velocity'])
    IMUData = namedtuple('IMUData', ['accelerometer', 'gyroscope'])

    def __init__(self, args):
        super(MidAirDataset, self).__init__(args)

        # Training datasets - sunny trajectories 0000-0019
        self.datasets_train_filter["sunny_sensor_records_trajectory_0000"] = [0, 8818]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0001"] = [0, 8820]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0002"] = [0, 8805]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0003"] = [0, 8817]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0004"] = [0, 8928]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0005"] = [0, 8821]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0006"] = [0, 8822]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0007"] = [0, 8813]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0008"] = [0, 8820]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0009"] = [0, 8833]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0010"] = [0, 8828]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0011"] = [0, 8834]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0012"] = [0, 8819]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0013"] = [0, 8823]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0014"] = [0, 8843]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0015"] = [0, 8798]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0016"] = [0, 8798]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0017"] = [0, 8808]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0018"] = [0, 8828]
        self.datasets_train_filter["sunny_sensor_records_trajectory_0019"] = [0, 8824]

        # Validation datasets - sunny trajectories 0020-0029
        self.datasets_validatation_filter["sunny_sensor_records_trajectory_0020"] = [0, 8818]
        self.datasets_validatation_filter["sunny_sensor_records_trajectory_0021"] = [0, 8813]
        self.datasets_validatation_filter["sunny_sensor_records_trajectory_0022"] = [0, 8811]
        self.datasets_validatation_filter["sunny_sensor_records_trajectory_0023"] = [0, 8820]
        self.datasets_validatation_filter["sunny_sensor_records_trajectory_0024"] = [0, 9022]
        self.datasets_validatation_filter["sunny_sensor_records_trajectory_0025"] = [0, 8819]
        self.datasets_validatation_filter["sunny_sensor_records_trajectory_0026"] = [0, 8813]
        self.datasets_validatation_filter["sunny_sensor_records_trajectory_0027"] = [0, 8818]
        self.datasets_validatation_filter["sunny_sensor_records_trajectory_0028"] = [0, 8819]
        self.datasets_validatation_filter["sunny_sensor_records_trajectory_0029"] = [0, 8818]

    @staticmethod
    def read_data(args):
        """
        Read the data from the MidAir dataset and save in the same format
        as the KITTI read_data: t (s), u (6-vector), ang_gt, p_gt, v_gt, name,
        t0.
        """
        print("Start read_data for MidAir dataset")
        
        # Find all HDF5 files in the data directory
        hdf5_files = []
        for root, dirs, files in os.walk(args.path_data_base):
            for file in files:
                if file.endswith('.hdf5'):
                    hdf5_files.append(os.path.join(root, file))
        
        print(f"Found {len(hdf5_files)} HDF5 files")
        
        t_tot = 0  # sum of times for the all dataset
        
        for file_path in hdf5_files:
            print(f"Processing file: {file_path}")
            
            try:
                with h5py.File(file_path, 'r') as f:
                    # Get all trajectory keys
                    trajectory_keys = [key for key in f.keys() if key.startswith('trajectory_')]
                    
                    for traj_key in trajectory_keys:
                        traj = f[traj_key]
                        
                        # Extract dataset name from file path and trajectory
                        file_name = os.path.splitext(os.path.basename(file_path))[0]
                        parent_dir = os.path.basename(os.path.dirname(file_path))
                        dataset_name = f"{parent_dir}_{file_name}_{traj_key}"
                        
                        print(f"Processing trajectory: {dataset_name}")
                        
                        # Load IMU data
                        imu_data = MidAirDataset.load_imu_data_from_group(traj['imu'])
                        
                        # Load ground truth data
                        gt_data = MidAirDataset.load_ground_truth_from_group(traj['groundtruth'])
                        
                        # Check data consistency
                        n_samples = len(imu_data.accelerometer)
                        if (len(imu_data.gyroscope) != n_samples or 
                            len(gt_data.position) != n_samples or
                            len(gt_data.attitude) != n_samples or
                            len(gt_data.velocity) != n_samples):
                            cprint(f"Data length mismatch in {dataset_name}", 'red')
                            continue
                        
                        # Skip if trajectory is too short
                        if n_samples < 100:
                            cprint(f"Trajectory {dataset_name} too short ({n_samples} samples), skipping", 'yellow')
                            continue
                        
                        # Create time vector (MidAir dataset is typically at 100Hz)
                        dt = 0.01  # 100 Hz
                        t = np.arange(n_samples) * dt
                        t0 = 0.0  # Start time
                        
                        # Convert quaternions to roll, pitch, yaw
                        ang_gt = np.zeros((n_samples, 3))
                        for i in range(n_samples):
                            quat = gt_data.attitude[i]  # quaternion format
                            # Convert quaternion to rotation matrix then to Euler angles
                            R = quat_to_rotation_matrix(quat)
                            roll, pitch, yaw = to_rpy(R)
                            ang_gt[i] = [roll, pitch, yaw]
                        
                        # Combine gyro and accelerometer data (gyro first, then acc like KITTI)
                        u_arr = np.concatenate([imu_data.gyroscope, imu_data.accelerometer], axis=1)
                        
                        # Convert to torch tensors
                        t = torch.from_numpy(t).float()
                        u_t = torch.from_numpy(u_arr).float()
                        p_gt = torch.from_numpy(gt_data.position).float()
                        v_gt = torch.from_numpy(gt_data.velocity).float()
                        ang_gt = torch.from_numpy(ang_gt).float()

                        mondict = {
                            't': t, # [N] time from start (s)
                            'u': u_t, # [N * 6] imu inputs gyro+acc
                            'ang_gt': ang_gt, # [N * 3] roll,pitch,heading (rad)
                            'p_gt': p_gt, # [N * 3] position (m)
                            'v_gt': v_gt, # [N * 3] velocities (m/s)
                            'name': dataset_name,
                            't0': t0, # original first IMU time
                        }
                        
                        # Update total time
                        t_tot += t[-1] - t[0]
                        
                        BaseDataset.dump(mondict, args.path_data_save, dataset_name)
                        print(f"Saved dataset: {dataset_name} - Duration: {t[-1]:.2f}s, Samples: {n_samples}")
                        
            except Exception as e:
                cprint(f"Error processing file {file_path}: {str(e)}", 'red')
                continue
        
        print(f"\nTotal dataset duration: {t_tot:.2f} s")

    @staticmethod
    def load_imu_data_from_group(imu_group) -> IMUData:
        """
        Load IMU data from HDF5 group.
        """
        accelerometer = np.array(imu_group['accelerometer'])
        gyroscope = np.array(imu_group['gyroscope'])
        
        return MidAirDataset.IMUData(
            accelerometer=accelerometer,
            gyroscope=gyroscope
        )
    
    @staticmethod
    def load_ground_truth_from_group(gt_group) -> GroundTruth:
        """
        Load ground truth data from HDF5 group.
        """
        position = np.array(gt_group['position'])
        attitude = np.array(gt_group['attitude'])  # quaternions [w, x, y, z]
        velocity = np.array(gt_group['velocity'])
        acceleration = np.array(gt_group['acceleration'])
        angular_velocity = np.array(gt_group['angular_velocity'])
        
        return MidAirDataset.GroundTruth(
            position=position,
            attitude=attitude,
            velocity=velocity,
            acceleration=acceleration,
            angular_velocity=angular_velocity
        )

    @staticmethod
    def load_imu_data(path: str) -> list[IMUData]:
        """
        Load IMU data from the specified path, with the quality specified by IMU_QUALITY.
        """
        # This method can be used for loading specific trajectory data if needed
        with h5py.File(path, 'r') as f:
            trajectory_keys = [key for key in f.keys() if key.startswith('trajectory_')]
            imu_datas = []
            
            for traj_key in trajectory_keys:
                imu_data = MidAirDataset.load_imu_data_from_group(f[traj_key]['imu'])
                imu_datas.append(imu_data)

        print(f"Loaded {len(imu_datas)} IMU data trajectories from {path}")
        return imu_datas
    
    @staticmethod
    def load_ground_truth(path: str) -> list[GroundTruth]:
        """
        Load ground truth data from the specified path.
        """
        # This method can be used for loading specific trajectory data if needed
        with h5py.File(path, 'r') as f:
            trajectory_keys = [key for key in f.keys() if key.startswith('trajectory_')]
            gt_data = []
            
            for traj_key in trajectory_keys:
                gt = MidAirDataset.load_ground_truth_from_group(f[traj_key]['groundtruth'])
                gt_data.append(gt)

        print(f"Loaded {len(gt_data)} ground truth trajectories from {path}")
        return gt_data