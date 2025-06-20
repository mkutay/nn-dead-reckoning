import torch
import time
import sys

from torch_iekf import TORCHIEKF # type: ignore
from numpy_iekf import NUMPYIEKF # type: ignore
from utils import prepare_data
from args import Args
from plot import IEKF_plot # type: ignore
from train_torch import train_filter as train_filter_cpu # type: ignore
from train_torch_gpu import train_filter as train_filter_gpu # type: ignore

args = Args()
numpy_iekf = NUMPYIEKF(args.parameter_class)
torch_iekf = TORCHIEKF(args.parameter_class)
dataset = args.dataset_class(args)

do = sys.argv[1]
if do == "read":
    dataset.read_data(args)
elif do == "train-gpu":
    train_filter_gpu(args, dataset)
elif do == "train-cpu":
    train_filter_cpu(args, dataset)
elif do == "filter":
    torch_iekf.load(args, dataset)
    numpy_iekf.set_learned_covariance(torch_iekf)

    for i in range(0, len(dataset.datasets)):
        dataset_name = dataset.dataset_name(i)
        if "sunny" not in dataset_name:
            continue

        print("Test filter on sequence: " + dataset_name)

        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, i, to_numpy=True)
        N = None
        u_t = torch.from_numpy(u).double()

        start_net_time = time.time()
        measurements_covs = torch_iekf.forward_nets(u_t).detach().numpy()
        net_time = time.time() - start_net_time
        print("Neural network inference time: {:.4f} s".format(net_time))

        start_time = time.time()
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = numpy_iekf.run(t, u, measurements_covs, v_gt, p_gt, N, ang_gt[0])
        diff_time = time.time() - start_time
        print("Execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time, t[-1] - t[0]))

        mondict = {
            't': t,
            'Rot': Rot,
            'v': v,
            'p': p,
            'b_omega': b_omega,
            'b_acc': b_acc,
            'Rot_c_i': Rot_c_i,
            't_c_i': t_c_i,
            'measurements_covs': measurements_covs,
        }
        
        dataset.dump(mondict, args.path_results, dataset_name + "_filter.p")

        print()
elif do == "plot":
    IEKF_plot(args, dataset)
else:
    print("Unknown command. Use 'read', 'filter', 'train', or 'plot'.")
    sys.exit(1)