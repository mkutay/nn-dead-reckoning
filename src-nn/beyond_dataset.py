# type: ignore

from collections import namedtuple
from termcolor import cprint
import numpy as np
import torch
import os

from base_dataset import BaseDataset
from utils import *

class BeyondDataset(BaseDataset):
    IMU_TYPE = "isense_kalman" # isense_kalman, isense_raw, yostlab

    # Bundle into an easy-to-access structure
    GroundTruth = namedtuple('GroundTruth', ['timestamp', 'frame_file', 'easting', 'northing', 'height', 'heading'])
    IMUData = namedtuple('IMUData', ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'heading'])

    def __init__(self, args):
        super(BeyondDataset, self).__init__(args)

        self.datasets_validatation_filter['Mamak_08_08'] = [0, 15142]
        self.datasets_train_filter["Mamak_08_08_ucus2"] = [0, 40811]
        self.datasets_train_filter["Mamak_21_07"] = [0, 148000]
        self.datasets_train_filter["Mamak_22_07"] = [0, 200000]

    @staticmethod
    def read_data(args):
        """
        Read the data from the GREAT IMU dataset and save in the same format
        as the KITTI read_data: t (s), u (6-vector), ang_gt, p_gt, v_gt, name,
        t0.
        """
        print("Start read_data for IMU dataset")
        data_dirs = os.listdir(args.path_data_base)
        for data_dir in data_dirs:
            path = os.path.join(args.path_data_base, data_dir)
            if not os.path.isdir(path):
                continue

            # Load ground truth and IMU
            path_gt = os.path.join(path, 'groundtruth.csv')
            gt_data = BeyondDataset.load_ground_truth(path_gt)
            imu_data = BeyondDataset.load_imu_data(path)
            if not gt_data or not imu_data:
                cprint(f"Skipping {data_dir} due to missing data.", 'red')
                continue

            print(f"\nProcessing sequence: {data_dir}")
            dataset_name = data_dir

            # Extract and align timestamps
            gps_times = np.array([gt.timestamp for gt in gt_data], dtype=np.float64)
            imu_times = np.array([imu.timestamp for imu in imu_data], dtype=np.float64)

            # Reference time
            t0 = imu_times[0]
            t = imu_times - t0
            gps_times -= t0

            # Nearest-neighbor sync of GT (25Hz) to IMU (200Hz)
            # For each imu timestamp find closest gt index
            indices = np.argmin(np.abs(gps_times[None, :] - t[:, None]), axis=1)

            # Build synchronized arrays
            gyro = np.vstack([[imu.gyro_x, imu.gyro_y, imu.gyro_z] for imu in imu_data])
            accel = np.vstack([[imu.accel_x, imu.accel_y, imu.accel_z * -1] for imu in imu_data])
            u_arr = np.hstack((gyro, accel))

            # Ground truth position (e, n, u) and subtract initial fix
            easting = np.array([gt.easting for gt in gt_data], dtype=np.float64)
            northing = np.array([gt.northing for gt in gt_data], dtype=np.float64)
            height = np.array([gt.height for gt in gt_data], dtype=np.float64)
            easting_i = easting[indices]
            northing_i = northing[indices]
            height_i = height[indices]
            p_arr = np.vstack((easting_i, northing_i, height_i)).T
            p_arr -= p_arr[0] # now relative to first GPS point

            # Ground truth velocity [vx, vy, vz]
            v_arr = np.zeros((len(indices), 3))

            # Ground truth angles: [roll, pitch, heading] in radians
            roll = np.zeros(len(gt_data), dtype=np.float64)
            pitch = np.zeros(len(gt_data), dtype=np.float64)
            heading = np.array([gt.heading for gt in gt_data], dtype=np.float64)
            # heading *= -1
            # heading += 90
            heading = np.where(heading > 180, heading - 360, heading)
            heading = np.where(heading < -180, heading + 360, heading)
            ang_arr = np.vstack((roll[indices], pitch[indices], heading[indices])).T * np.pi / 180.0

            t = torch.from_numpy(t).float()
            u_t = torch.from_numpy(u_arr).float()
            p_gt = torch.from_numpy(p_arr).float()
            v_gt = torch.from_numpy(v_arr).float()
            ang_gt = torch.from_numpy(ang_arr).float()

            mondict = {
                't': t, # [N] time from start (s)
                'u': u_t, # [N * 6] imu inputs gyro+acc
                'ang_gt': ang_gt, # [N * 3] roll,pitch,heading (rad)
                'p_gt': p_gt, # [N * 3] easting,northing,up (m)
                'v_gt': v_gt, # [N * 3] velocities (m/s)
                'name': dataset_name,
                't0': t0, # original first IMU time
            }
            BaseDataset.dump(mondict, args.path_data_save, dataset_name)
            print(f"Saved dataset: {dataset_name}")

    @staticmethod
    def load_imu_data(path: str) -> list[IMUData]:
        """
        Load IMU data from the specified path, with the quality specified by IMU_QUALITY.
        """
        imu_path = os.path.join(path, 'imu_' + BeyondDataset.IMU_TYPE + '.csv')
        imu_datas = []
        
        if not os.path.exists(imu_path):
            cprint(f"IMU data file not found: {imu_path}", 'red')
            return imu_datas
            
        with open(imu_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                parts = line.split(",")
                if len(parts) < 7:
                    cprint(f"Skipping invalid line in {imu_path}: {line}")
                    continue

                try:
                    imu_data = BeyondDataset.IMUData(
                        timestamp=float(parts[0]),
                        gyro_x=float(parts[1]),
                        gyro_y=float(parts[2]),
                        gyro_z=float(parts[3]),
                        accel_x=float(parts[4]),
                        accel_y=float(parts[5]),
                        accel_z=float(parts[6]),
                        heading=float(parts[7]) if len(parts) > 7 else 0.0 # Optional heading, default to 0.0 if not present
                    )
                    imu_datas.append(imu_data)
                except TypeError as e:
                    cprint(f"Error parsing line in {imu_path}: {line} - {e}", 'red')
                    continue

        print(f"Loaded {len(imu_datas)} IMU data points from {imu_path}")
        return imu_datas
    
    @staticmethod
    def load_ground_truth(path: str) -> list[GroundTruth]:
        """
        Load ground truth data from the specified path.
        """
        gt_data = []
        
        if not os.path.exists(path):
            cprint(f"Ground truth file not found: {path}", 'red')
            return gt_data
            
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                parts = line.split(",")
                if len(parts) < 6:
                    cprint(f"Skipping invalid line in {path}: {line}")
                    continue

                try:
                    gt = BeyondDataset.GroundTruth(
                        timestamp=float(parts[0]),
                        frame_file=parts[1],
                        easting=float(parts[2]),
                        northing=float(parts[3]),
                        height=parts[4],
                        heading=float(parts[5])
                    )
                    gt_data.append(gt)
                except ValueError as e:
                    cprint(f"Error parsing line in {path}: {line} - {e}", 'red')
                    continue

        print(f"Loaded {len(gt_data)} ground truth data points from {path}")
        return gt_data