# type: ignore

from collections import namedtuple, OrderedDict
from torch.utils.data.dataset import Dataset
from termcolor import cprint
import numpy as np
import datetime
import torch
import glob
import os

from numpy_iekf import NUMPYIEKF
from utils import *
from base_dataset import BaseDataset

class KITTIDataset(BaseDataset):
    # Bundle into an easy-to-access structure
    OxtsPacket = namedtuple('OxtsPacket', ['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vf', 'vl', 'vu', 'ax', 'ay', 'az', 'af', 'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'pos_accuracy', 'vel_accuracy', 'navstat', 'numsats', 'posmode', 'velmode', 'orimode'])
    OxtsData = namedtuple('OxtsData', ['packet', 'T_w_imu'])

    def __init__(self, args):
        super(KITTIDataset, self).__init__(args)

        self.datasets_validatation_filter['2011_09_30_drive_0028_extract'] = [11231, 53650]
        self.datasets_train_filter["2011_10_03_drive_0042_extract"] = [0, None]
        self.datasets_train_filter["2011_09_30_drive_0018_extract"] = [0, 15000]
        self.datasets_train_filter["2011_09_30_drive_0020_extract"] = [0, None]
        self.datasets_train_filter["2011_09_30_drive_0027_extract"] = [0, None]
        self.datasets_train_filter["2011_09_30_drive_0033_extract"] = [0, None]
        self.datasets_train_filter["2011_10_03_drive_0027_extract"] = [0, 18000]
        self.datasets_train_filter["2011_10_03_drive_0034_extract"] = [0, 31000]
        self.datasets_train_filter["2011_09_30_drive_0034_extract"] = [0, None]

    @staticmethod
    def read_data(args):
        """
        Read the data from the KITTI dataset
        """

        print("Start read_data")

        t_tot = 0 # sum of times for the all dataset
        data_dirs = os.listdir(args.path_data_base)
        for n_iteration, date_dir in enumerate(data_dirs):
            # get access to each sequence
            path = os.path.join(args.path_data_base, date_dir)
            if not os.path.isdir(path):
                continue

            # read data
            oxts_files = sorted(glob.glob(os.path.join(path, 'data', '*.txt')))
            oxts = KITTIDataset.load_oxts_packets_and_poses(oxts_files)

            """
            Note on difference between ground truth and oxts solution:
            - orientation is the same
            - north and east axis are inverted
            - position are closed to but different
            => oxts solution is not loaded
            """

            print("\n Sequence name : " + date_dir)

            lat_oxts = np.zeros(len(oxts))
            lon_oxts = np.zeros(len(oxts))
            alt_oxts = np.zeros(len(oxts))
            roll_oxts = np.zeros(len(oxts))
            pitch_oxts = np.zeros(len(oxts))
            yaw_oxts = np.zeros(len(oxts))
            roll_gt = np.zeros(len(oxts))
            pitch_gt = np.zeros(len(oxts))
            yaw_gt = np.zeros(len(oxts))
            t = KITTIDataset.load_timestamps(path)
            acc = np.zeros((len(oxts), 3))
            acc_bis = np.zeros((len(oxts), 3))
            gyro = np.zeros((len(oxts), 3))
            gyro_bis = np.zeros((len(oxts), 3))
            p_gt = np.zeros((len(oxts), 3))
            v_gt = np.zeros((len(oxts), 3))
            v_rob_gt = np.zeros((len(oxts), 3))

            k_max = len(oxts)
            for k in range(k_max):
                oxts_k = oxts[k]
                t[k] = 3600 * t[k].hour + 60 * t[k].minute + t[k].second + t[k].microsecond / 1e6
                lat_oxts[k] = oxts_k[0].lat
                lon_oxts[k] = oxts_k[0].lon
                alt_oxts[k] = oxts_k[0].alt
                acc[k, 0] = oxts_k[0].af
                acc[k, 1] = oxts_k[0].al
                acc[k, 2] = oxts_k[0].au
                acc_bis[k, 0] = oxts_k[0].ax
                acc_bis[k, 1] = oxts_k[0].ay
                acc_bis[k, 2] = oxts_k[0].az
                gyro[k, 0] = oxts_k[0].wf
                gyro[k, 1] = oxts_k[0].wl
                gyro[k, 2] = oxts_k[0].wu
                gyro_bis[k, 0] = oxts_k[0].wx
                gyro_bis[k, 1] = oxts_k[0].wy
                gyro_bis[k, 2] = oxts_k[0].wz
                roll_oxts[k] = oxts_k[0].roll
                pitch_oxts[k] = oxts_k[0].pitch
                yaw_oxts[k] = oxts_k[0].yaw
                v_gt[k, 0] = oxts_k[0].ve
                v_gt[k, 1] = oxts_k[0].vn
                v_gt[k, 2] = oxts_k[0].vu
                v_rob_gt[k, 0] = oxts_k[0].vf
                v_rob_gt[k, 1] = oxts_k[0].vl
                v_rob_gt[k, 2] = oxts_k[0].vu
                p_gt[k] = oxts_k[1][:3, 3]
                Rot_gt_k = oxts_k[1][:3, :3]
                roll_gt[k], pitch_gt[k], yaw_gt[k] = to_rpy(Rot_gt_k)

            t0 = t[0]
            t = np.array(t) - t[0]

            # some data can have gps out
            if np.max(t[:-1] - t[1:]) > 0.1:
                cprint(date_dir2 + " has time problem", 'yellow')

            ang_gt = np.zeros((roll_gt.shape[0], 3))
            ang_gt[:, 0] = roll_gt
            ang_gt[:, 1] = pitch_gt
            ang_gt[:, 2] = yaw_gt

            # take correct imu measurements
            u = np.concatenate((gyro_bis, acc_bis), -1)

            # convert from numpy to torch and float
            t = torch.from_numpy(t).float()
            p_gt = torch.from_numpy(p_gt).float()
            v_gt = torch.from_numpy(v_gt).float()
            ang_gt = torch.from_numpy(ang_gt).float()
            u = torch.from_numpy(u).float()

            mondict = {
                't': t, # time vector
                'p_gt': p_gt, # ground truth position
                'ang_gt': ang_gt, # ground truth angles: roll, pitch, yaw
                'v_gt': v_gt, # ground truth velocity
                'u': u, # imu measurements: gyro, acc
                'name': date_dir, # name of the sequence
                't0': t0, # initial time
            }

            t_tot += t[-1] - t[0]
            BaseDataset.dump(mondict, args.path_data_save, date_dir)
            
        print("\n Total dataset duration : {:.2f} s".format(t_tot))
    
    @staticmethod
    def load_timestamps(data_path):
        """Load timestamps from file."""
        timestamp_file = os.path.join(data_path, 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                t = datetime.datetime.fromisoformat(line.strip())
                timestamps.append(t)
        return timestamps

    @staticmethod
    def pose_from_oxts_packet(packet, scale):
        """Helper method to compute a SE(3) pose matrix from an OXTS packet."""
        earth_radius = 6378137. # earth radius (approx.) in meters

        # Use a Mercator projection to get the *translation vector*
        tx = scale * packet.lon * np.pi * earth_radius / 180.
        ty = scale * earth_radius * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])
    
        # Use the Euler angles to get the *rotation matrix*
        Rx = rotx(packet.roll)
        Ry = roty(packet.pitch)
        Rz = rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        return R, t

    @staticmethod
    def transform_from_rot_trans(R, t):
        """Transformation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    @staticmethod
    def load_oxts_packets_and_poses(oxts_files: list[str]) -> list[OxtsData]:
        """
        Generator to read OXTS ground truth data.
        Poses are given in an East-North-Up coordinate system,
        whose origin is the first GPS position.
        """
        # Scale for Mercator projection (from first lat value)
        scale = None
        # Origin of the global coordinate system (first GPS position)
        origin = None

        oxts = []

        for filename in oxts_files:
            with open(filename, 'r') as f:
                lines = f.readlines()
                line = lines[0].strip().split() # there should only be one line per file

                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = KITTIDataset.OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = KITTIDataset.pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t

                T_w_imu = KITTIDataset.transform_from_rot_trans(R, t - origin)

                oxts.append(KITTIDataset.OxtsData(packet, T_w_imu))

        return oxts