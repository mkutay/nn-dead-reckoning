# type: ignore

from collections import namedtuple
from termcolor import cprint
import numpy as np
import torch
import os

from base_dataset import BaseDataset
from utils import *

class GreatDataset(BaseDataset):
    IMU_QUALITY = "Tactical" # IMU quality, can be "MEMS" or "Tactical"

    # Bundle into an easy-to-access structure
    GroundTruth = namedtuple('GroundTruth', ['weeks', 'gps_time', 'e', 'n', 'u', 'vx', 'vy', 'vz', 'heading', 'pitch', 'roll'])
    IMUData = namedtuple('IMUData', ['time', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z'])

    def __init__(self, args):
        super(GreatDataset, self).__init__(args)

    @staticmethod
    def read_data(args):
        print("Start read_data for IMU dataset")

        data_dirs = os.listdir(args.path_data_base)
        for n_iteration, data_dir in enumerate(data_dirs):
            # get access to each sequence
            path = os.path.join(args.path_data_base, data_dir)
            if not os.path.isdir(path):
                continue

            # Load ground truth
            path_gt = os.path.join(path, 'groundtruth.txt')
            ground_truth_data = GreatDataset.load_ground_truth(path_gt)
            imu_data = GreatDataset.load_imu_data(path)

            if not ground_truth_data or not imu_data:
                cprint(f"Skipping {data_dir} due to missing data.", 'red')
                continue

            print(f"\nProcessing sequence: {data_dir}")
            
            dataset_name = data_dir

            # Extract ground truth data
            gps_time_gt = np.array([gt.gps_time for gt in ground_truth_data], dtype=np.float64)
            e_gt = np.array([gt.e for gt in ground_truth_data], dtype=np.float64)
            n_gt = np.array([gt.n for gt in ground_truth_data], dtype=np.float64)
            u_gt = np.array([gt.u for gt in ground_truth_data], dtype=np.float64)
            vx_gt = np.array([gt.vx for gt in ground_truth_data], dtype=np.float64)
            vy_gt = np.array([gt.vy for gt in ground_truth_data], dtype=np.float64)
            vz_gt = np.array([gt.vz for gt in ground_truth_data], dtype=np.float64)
            heading_gt = np.array([gt.heading for gt in ground_truth_data], dtype=np.float64)
            pitch_gt = np.array([gt.pitch for gt in ground_truth_data], dtype=np.float64)
            roll_gt = np.array([gt.roll for gt in ground_truth_data], dtype=np.float64)

            # Extract IMU data
            time_imu = np.array([float(imu.time) for imu in imu_data], dtype=np.float64)
            gyro_x = np.array([float(imu.gyro_x) for imu in imu_data], dtype=np.float64)
            gyro_y = np.array([float(imu.gyro_y) for imu in imu_data], dtype=np.float64)
            gyro_z = np.array([float(imu.gyro_z) for imu in imu_data], dtype=np.float64)
            accel_x = np.array([float(imu.accel_x) for imu in imu_data], dtype=np.float64)
            accel_y = np.array([float(imu.accel_y) for imu in imu_data], dtype=np.float64)
            accel_z = np.array([float(imu.accel_z) for imu in imu_data], dtype=np.float64)

            t0 = time_imu[0]
            time = time_imu - t0
            
            # Adjust ground truth time to same reference
            gps_time_adjusted = gps_time_gt - t0

            # Synchronize ground truth to IMU timestamps using nearest neighbor interpolation
            print(f"Synchronizing ground truth (10Hz) to IMU timestamps (100Hz)...")
            
            # Find closest ground truth for each IMU timestamp
            closest_indices = []
            for imu_time in time:
                # Find the index of the closest ground truth timestamp
                time_diffs = np.abs(gps_time_adjusted - imu_time)
                closest_idx = np.argmin(time_diffs)
                closest_indices.append(closest_idx)
            
            closest_indices = np.array(closest_indices)
            
            # Interpolate ground truth data to match IMU timestamps
            e_interp = e_gt[closest_indices]
            n_interp = n_gt[closest_indices]
            u_interp = u_gt[closest_indices]
            vx_interp = vx_gt[closest_indices]
            vy_interp = vy_gt[closest_indices]
            vz_interp = vz_gt[closest_indices]
            heading_interp = heading_gt[closest_indices]
            pitch_interp = pitch_gt[closest_indices]
            roll_interp = roll_gt[closest_indices]

            # Prepare IMU data matrix
            gyro_bis = np.zeros((len(imu_data), 3))
            acc_bis = np.zeros((len(imu_data), 3))
            for k in range(len(imu_data)):
                gyro_bis[k, 0] = gyro_x[k]
                gyro_bis[k, 1] = gyro_y[k]
                gyro_bis[k, 2] = gyro_z[k]
                acc_bis[k, 0] = accel_x[k]
                acc_bis[k, 1] = accel_y[k]
                acc_bis[k, 2] = accel_z[k]
            u_t = np.concatenate((gyro_bis, acc_bis), -1)

            # Prepare ground truth matrices (now synchronized to IMU timestamps)
            v_gt = np.zeros((len(imu_data), 3))
            for k in range(len(imu_data)):
                v_gt[k, 0] = vx_interp[k]
                v_gt[k, 1] = vy_interp[k]
                v_gt[k, 2] = vz_interp[k]

            ang_gt = np.zeros((len(imu_data), 3))
            for k in range(len(imu_data)):
                ang_gt[k, 0] = roll_interp[k]
                ang_gt[k, 1] = pitch_interp[k]
                ang_gt[k, 2] = heading_interp[k]

            p_gt = np.zeros((len(imu_data), 3))
            for k in range(len(imu_data)):
                p_gt[k, 0] = e_interp[k]
                p_gt[k, 1] = n_interp[k]
                p_gt[k, 2] = u_interp[k]

            print(f"IMU data length: {len(imu_data)}")
            print(f"Ground truth data length (original): {len(ground_truth_data)}")
            print(f"Ground truth data length (synchronized): {len(p_gt)}")

            time = torch.from_numpy(time).float()
            u_t = torch.from_numpy(u_t).float()
            ang_gt = torch.from_numpy(ang_gt).float()
            p_gt = torch.from_numpy(p_gt).float()
            v_gt = torch.from_numpy(v_gt).float()

            # Save to pickle
            pickle_dict = {
                't': time,
                'u': u_t,
                'ang_gt': ang_gt,
                'p_gt': p_gt,
                'v_gt': v_gt,
                'name': dataset_name,
                't0': t0,
            }

            BaseDataset.dump(pickle_dict, args.path_data_save, dataset_name)
            print(f"Saved dataset: {dataset_name}")

    @staticmethod
    def load_imu_data(path: str) -> list[IMUData]:
        """
        Load IMU data from the specified path, with the quality specified by IMU_QUALITY.
        """
        imu_path = os.path.join(path, 'IMU', GreatDataset.IMU_QUALITY + '_imu_data.txt')
        imu_datas = []
        
        if not os.path.exists(imu_path):
            cprint(f"IMU data file not found: {imu_path}", 'red')
            return imu_datas
            
        with open(imu_path, 'r') as f:
            lines = f.readlines()
            # Skip first header line
            for line in lines[1:]:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                parts = line.split()
                if len(parts) < 7:
                    cprint(f"Skipping invalid line in {imu_path}: {line}")
                    continue

                try:
                    imu_data = GreatDataset.IMUData(
                        time=float(parts[0]),
                        gyro_x=float(parts[1]),
                        gyro_y=float(parts[2]),
                        gyro_z=float(parts[3]),
                        accel_x=float(parts[4]),
                        accel_y=float(parts[5]),
                        accel_z=float(parts[6])
                    )
                    imu_datas.append(imu_data)
                except ValueError as e:
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
            # Skip first two header lines
            for line in lines[2:]:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                parts = line.split()
                if len(parts) < 11:
                    cprint(f"Skipping invalid line in {path}: {line}")
                    continue

                try:
                    gt = GreatDataset.GroundTruth(
                        weeks=float(parts[0]),
                        gps_time=float(parts[1]),
                        e=float(parts[2]),
                        n=float(parts[3]),
                        u=float(parts[4]),
                        vx=float(parts[5]),
                        vy=float(parts[6]),
                        vz=float(parts[7]),
                        heading=float(parts[8]),
                        pitch=float(parts[9]),
                        roll=float(parts[10])
                    )
                    gt_data.append(gt)
                except ValueError as e:
                    cprint(f"Error parsing line in {path}: {line} - {e}", 'red')
                    continue

        print(f"Loaded {len(gt_data)} ground truth data points from {path}")
        return gt_data