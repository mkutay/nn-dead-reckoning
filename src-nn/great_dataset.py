# type: ignore

from collections import namedtuple
from termcolor import cprint
import numpy as np
import torch
import os

from base_dataset import BaseDataset
from utils import *

class GreatDataset(BaseDataset):
    IMU_QUALITY = "MEMS" # IMU quality, can be "MEMS" or "Tactical"

    # Bundle into an easy-to-access structure
    GroundTruth = namedtuple('GroundTruth', ['weeks', 'gps_time', 'e', 'n', 'u', 'vx', 'vy', 'vz', 'heading', 'pitch', 'roll'])
    IMUData = namedtuple('IMUData', ['time', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z'])

    def __init__(self, args):
        super(GreatDataset, self).__init__(args)

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
            path_gt = os.path.join(path, 'groundtruth.txt')
            gt_data = GreatDataset.load_ground_truth(path_gt)
            imu_data = GreatDataset.load_imu_data(path)
            if not gt_data or not imu_data:
                cprint(f"Skipping {data_dir} due to missing data.", 'red')
                continue

            print(f"\nProcessing sequence: {data_dir}")
            dataset_name = data_dir

            # Extract and align timestamps
            gps_times = np.array([gt.gps_time for gt in gt_data], dtype=np.float64)
            imu_times = np.array([imu.time for imu in imu_data], dtype=np.float64)

            # Reference time
            t0 = imu_times[0]
            t = imu_times - t0
            gps_times -= t0

            # Nearest-neighbor sync of GT (10Hz) to IMU (100Hz)
            # For each imu timestamp find closest gt index
            indices = np.argmin(np.abs(gps_times[None, :] - t[:, None]), axis=1)

            # Build synchronized arrays
            gyro = np.vstack([[imu.gyro_x, imu.gyro_y, imu.gyro_z] for imu in imu_data])
            accel = np.vstack([[imu.accel_x, imu.accel_y, imu.accel_z] for imu in imu_data])
            u_arr = np.hstack((gyro, accel))

            # Ground truth position (e, n, u) and subtract initial fix
            e = np.array([gt.e for gt in gt_data], dtype=np.float64)
            n = np.array([gt.n for gt in gt_data], dtype=np.float64)
            u_ = np.array([gt.u for gt in gt_data], dtype=np.float64)
            e_i = e[indices]
            n_i = n[indices]
            u_i = u_[indices]
            p_arr = np.vstack((e_i, n_i, u_i)).T
            p_arr -= p_arr[0] # now relative to first GPS point

            # Ground truth velocity [vx, vy, vz]
            vx = np.array([gt.vx for gt in gt_data], dtype=np.float64)
            vy = np.array([gt.vy for gt in gt_data], dtype=np.float64)
            # vy *= -1
            vz = np.array([gt.vz for gt in gt_data], dtype=np.float64)
            v_arr = np.vstack((vx[indices], vy[indices], vz[indices])).T

            # Ground truth angles: [roll, pitch, heading] in radians
            roll = np.array([gt.roll for gt in gt_data], dtype=np.float64)
            pitch = np.array([gt.pitch for gt in gt_data], dtype=np.float64)
            head = np.array([gt.heading for gt in gt_data], dtype=np.float64)
            # head = head - np.pi / 2 # adjust heading to match IMU frame from kitti
            ang_arr = np.vstack((roll[indices], pitch[indices], head[indices])).T * np.pi / 180.0

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