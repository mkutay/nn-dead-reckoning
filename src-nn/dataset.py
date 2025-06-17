# type: ignore

from collections import namedtuple, OrderedDict
from torch.utils.data.dataset import Dataset
from termcolor import cprint
from navpy import lla2ned
import numpy as np
import datetime
import pickle
import torch
import glob
import os

from numpy_iekf import NUMPYIEKF
from utils import *

class BaseDataset(Dataset):
    # Bundle into an easy-to-access structure
    OxtsPacket = namedtuple('OxtsPacket', ['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vf', 'vl', 'vu', 'ax', 'ay', 'az', 'af', 'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'pos_accuracy', 'vel_accuracy', 'navstat', 'numsats', 'posmode', 'velmode', 'orimode'])
    OxtsData = namedtuple('OxtsData', ['packet', 'T_w_imu'])

    pickle_extension = ".p"
    """extension of the file saved in pickle format"""
    file_normalize_factor = "normalize_factors.p"
    """name of file for normalizing input"""

    def __init__(self, args):
        # paths
        self.path_data_save = args.path_data_save
        """path where data are saved"""
        self.path_results = args.path_results
        """path to the results"""
        self.path_temp = args.path_temp
        """path for temporary files"""

        # names of the sequences
        self.datasets = []
        """dataset names"""
        self.datasets_train = []
        """train datasets"""

        self.datasets_validatation_filter = OrderedDict()
        """Validation dataset with index for starting/ending"""
        self.datasets_train_filter = OrderedDict()
        """Validation dataset with index for starting/ending"""

        # number of training data points
        self.num_data = 0

        # factors for normalizing inputs
        self.normalize_factors = None
        self.get_datasets()
        self.set_normalize_factors()

    def __getitem__(self, i):
        mondict = self.load(self.path_data_save, self.datasets[i])
        return mondict

    def __len__(self):
        return len(self.datasets)

    def get_datasets(self):
        for dataset in os.listdir(self.path_data_save):
            self.datasets += [dataset[:-2]] # take just name, remove the ".p"
        self.divide_datasets()

    def divide_datasets(self):
        for dataset in self.datasets:
            self.datasets_train += [dataset]

    def dataset_name(self, i):
        return self.datasets[i]

    def get_data(self, i):
        pickle_dict = self[self.datasets.index(i) if type(i) != int else i]
        return pickle_dict['t'], pickle_dict['ang_gt'], pickle_dict['p_gt'], pickle_dict['v_gt'], pickle_dict['u']

    def set_normalize_factors(self):
        """
        Compute the normalizing factors for the input data
        and save them in a file.
        The factors are computed as follows:
        - mean of the input data
        - standard deviation of the input data
        """
        path_normalize_factor = os.path.join(self.path_temp, self.file_normalize_factor)

        # we set factors only if file does not exist
        if os.path.isfile(path_normalize_factor):
            pickle_dict = self.load(path_normalize_factor)
            self.normalize_factors = pickle_dict['normalize_factors']
            self.num_data = pickle_dict['num_data']
            return

        # first compute mean
        self.num_data = 0

        for i, dataset in enumerate(self.datasets_train):
            pickle_dict = self.load(self.path_data_save, dataset)
            u = pickle_dict['u']
            if i == 0:
                u_loc = u.sum(dim=0)
            else:
                u_loc += u.sum(dim=0)
            self.num_data += u.shape[0]
        u_loc = u_loc / self.num_data

        # second compute standard deviation
        for i, dataset in enumerate(self.datasets_train):
            pickle_dict = self.load(self.path_data_save, dataset)
            u = pickle_dict['u']
            if i == 0:
                u_std = ((u - u_loc) ** 2).sum(dim=0)
            else:
                u_std += ((u - u_loc) ** 2).sum(dim=0)
        u_std = (u_std / self.num_data).sqrt()

        self.normalize_factors = {
            'u_loc': u_loc,
            'u_std': u_std,
        }

        print('... ended computing normalizing factors')

        pickle_dict = {
            'normalize_factors': self.normalize_factors,
            'num_data': self.num_data
        }
        
        self.dump(pickle_dict, path_normalize_factor)

    def normalize(self, u):
        u_loc = self.normalize_factors["u_loc"]
        u_std = self.normalize_factors["u_std"]
        u_normalized = (u - u_loc) / u_std
        return u_normalized

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
            oxts = BaseDataset.load_oxts_packets_and_poses(oxts_files)

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
            t = BaseDataset.load_timestamps(path)
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

            p_oxts = lla2ned(lat_oxts, lon_oxts, alt_oxts, lat_oxts[0], lon_oxts[0], alt_oxts[0], latlon_unit='deg', alt_unit='m', model='wgs84')
            p_oxts[:, [0, 1]] = p_oxts[:, [1, 0]] # invert north and east axis, see note above

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

    @classmethod
    def load(cls, *_file_name):
        file_name = os.path.join(*_file_name)
        if not file_name.endswith(cls.pickle_extension):
            file_name += cls.pickle_extension
        with open(file_name, "rb") as file_pi:
            pickle_dict = pickle.load(file_pi)
        return pickle_dict

    @classmethod
    def dump(cls, mondict, *_file_name):
        file_name = os.path.join(*_file_name)
        if not file_name.endswith(cls.pickle_extension):
            file_name += cls.pickle_extension
        with open(file_name, "wb") as file_pi:
            pickle.dump(mondict, file_pi)

    def init_state_torch_filter(self, iekf):
        b_omega0 = torch.zeros(3).double()
        b_acc0 = torch.zeros(3).double()
        Rot_c_i0 = torch.eye(3).double()
        t_c_i0 = torch.zeros(3).double()
        return b_omega0, b_acc0, Rot_c_i0, t_c_i0  

    def get_estimates(self, dataset_name):
        dataset_name = self.datasets[dataset_name] if type(dataset_name) == int else dataset_name
        file_name = os.path.join(self.path_results, dataset_name + "_filter.p")
        if not os.path.exists(file_name):
            print('No result for ' + dataset_name)
            return
        mondict = self.load(file_name)
        Rot = mondict['Rot']
        v = mondict['v']
        p = mondict['p']
        b_omega = mondict['b_omega']
        b_acc = mondict['b_acc']
        Rot_c_i = mondict['Rot_c_i']
        t_c_i = mondict['t_c_i']
        measurements_covs = mondict['measurements_covs']
        return Rot, v, p , b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs
    
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

                packet = BaseDataset.OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = BaseDataset.pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t

                T_w_imu = BaseDataset.transform_from_rot_trans(R, t - origin)

                oxts.append(BaseDataset.OxtsData(packet, T_w_imu))

        return oxts