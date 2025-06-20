# type: ignore

from termcolor import cprint
import numpy as np
import torch
import time
import copy
import os

from torch_iekf import TORCHIEKF
from utils import prepare_data

# Training constants optimized for MidAir drone dataset
max_loss = 1e5          # Much higher max loss threshold for initial training convergence
max_grad_norm = 5e0     # Higher gradient norm for training stability with high initial losses
min_lr = 1e-6           # Lower minimum learning rate for fine-tuning
criterion = torch.nn.MSELoss(reduction="sum")

# Learning rates optimized for drone IMU characteristics - more conservative for stability
lr_initprocesscov_net = 1e-5    # Much lower initial process covariance learning rate
weight_decay_initprocesscov_net = 1e-8  # Weight decay for regularization

lr_mesnet = {
    'cov_net': 1e-5,    # Much lower learning rate for measurement covariance network
    'cov_lin': 5e-5,    # Lower linear layer learning rate for stability
}
weight_decay_mesnet = {
    'cov_net': 1e-7,    # Higher weight decay for covariance network regularization
    'cov_lin': 1e-8,    # Standard weight decay for linear layers
}

def compute_delta_p(Rot, p):
    """Compute delta position - GPU optimized version"""
    list_rpe = [[], [], []]  # [idx_0, idx_end, pose_delta_p]

    # sample at 1 Hz
    Rot = Rot[::10]
    p = p[::10]

    step_size = 10  # every second
    distances = torch.zeros(p.shape[0], device=p.device)
    dp = p[1:] - p[:-1]  #  this must be ground truth
    distances[1:] = dp.norm(dim=1).cumsum(0)

    seq_lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    k_max = int(Rot.shape[0] / step_size) - 1

    for k in range(0, k_max):
        idx_0 = k * step_size
        for seq_length in seq_lengths:
            if seq_length + distances[idx_0] > distances[-1]:
                continue
            idx_shift = torch.searchsorted(distances[idx_0:], distances[idx_0] + seq_length)
            idx_end = idx_0 + idx_shift

            list_rpe[0].append(idx_0)
            list_rpe[1].append(idx_end)

        idxs_0 = torch.tensor(list_rpe[0], device=Rot.device)
        idxs_end = torch.tensor(list_rpe[1], device=Rot.device)
        delta_p = Rot[idxs_0].transpose(-1, -2).matmul(
            ((p[idxs_end] - p[idxs_0]).float()).unsqueeze(-1)).squeeze()
        list_rpe[2] = delta_p
    return list_rpe


def train_filter(args, dataset):
    """Main training function - GPU version with Apple Silicon M-chip support"""
    # Enhanced GPU detection for Apple Silicon M-chips
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon GPU (MPS)"
        print(f"Using device: {device}")
        print(f"GPU: {device_name}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device: {device}")
        print(f"GPU: {device_name}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
        print("No GPU acceleration available")
    
    iekf = prepare_filter(args, dataset, device)
    prepare_loss_data(args, dataset, device)
    save_iekf(args, iekf)
    optimizer = set_optimizer(iekf)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loop(args, dataset, epoch, iekf, optimizer, args.seq_dim, device)
        save_iekf(args, iekf)
        
        # Memory management for different device types
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        elif device.type == 'mps':
            # MPS doesn't have detailed memory reporting like CUDA
            torch.mps.empty_cache()
            print("MPS cache cleared")
        
        print("Amount of time spent for 1 epoch: {}s\n".format(int(time.time() - start_time)))
        start_time = time.time()


def prepare_filter(args, dataset, device):
    """Prepare IEKF filter for GPU training with MPS float32 support"""
    print(f"Preparing IEKF filter for device: {device}")
    
    iekf = TORCHIEKF()

    # set dataset parameter
    iekf.filter_parameters = args.parameter_class()
    iekf.set_param_attr()
    
    # Handle gravity vector first (before any device operations)
    if type(iekf.g).__module__ == np.__name__:
        print("Converting gravity vector...")
        if device.type == 'mps':
            iekf.g = torch.from_numpy(iekf.g).float()
        else:
            iekf.g = torch.from_numpy(iekf.g).double()
        print(f"Gravity converted: {iekf.g.dtype}")

    # load model (this might load float64 weights)
    if args.continue_training:
        print("Loading existing model...")
        iekf.load(args, dataset)
    
    # CRITICAL: Convert ALL model parameters to appropriate precision BEFORE device transfer
    print("Converting model precision...")
    if device.type == 'mps':
        print("Converting entire model to float32...")
        # Recursively convert all parameters and buffers to float32
        def convert_to_float32(module):
            for child in module.children():
                convert_to_float32(child)
            for param in module.parameters(recurse=False):
                param.data = param.data.float()
            for buffer in module.buffers(recurse=False):
                buffer.data = buffer.data.float()
        
        convert_to_float32(iekf)
        
        # Handle gravity separately
        if hasattr(iekf, 'g') and iekf.g is not None:
            iekf.g = iekf.g.float()
    else:
        print("Converting entire model to float64...")
        # Recursively convert all parameters and buffers to float64
        def convert_to_float64(module):
            for child in module.children():
                convert_to_float64(child)
            for param in module.parameters(recurse=False):
                param.data = param.data.double()
            for buffer in module.buffers(recurse=False):
                buffer.data = buffer.data.double()
        
        convert_to_float64(iekf)
        
        # Handle gravity separately
        if hasattr(iekf, 'g') and iekf.g is not None:
            iekf.g = iekf.g.double()
    
    print("Moving model to device...")
    # Now move to device (all tensors should have compatible precision)
    iekf = iekf.to(device)
    if hasattr(iekf, 'g') and iekf.g is not None:
        iekf.g = iekf.g.to(device)
        print(f"Model moved to device. Gravity: {iekf.g.device}, {iekf.g.dtype}")
    
    iekf.train()
    
    # init u_loc and u_std
    print("Initializing normalization parameters...")
    iekf.get_normalize_u(dataset)
    print("IEKF filter prepared successfully!")
    return iekf

def prepare_loss_data(args, dataset, device):
    """Prepare loss data with GPU support and MPS float32 compatibility"""
    file_delta_p = os.path.join(args.path_temp, 'delta_p_gpu.p')
    if os.path.isfile(file_delta_p):
        mondict = dataset.load(file_delta_p)
        dataset.list_rpe = mondict['list_rpe']
        dataset.list_rpe_validation = mondict['list_rpe_validation']
        if set(dataset.datasets_train_filter.keys()) <= set(dataset.list_rpe.keys()): 
            return

    print("Preparing loss data on GPU...")
    
    # Determine precision based on device
    dtype = torch.float32 if device.type == 'mps' else torch.float64
    
    # prepare delta_p_gt
    list_rpe = {}
    for dataset_name, Ns in dataset.datasets_train_filter.items():
        print(f"Processing training dataset: {dataset_name}")
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, 0)
        
        if device.type == 'mps':
            p_gt = p_gt.float().to(device)
            ang_gt = ang_gt.float().to(device)
        else:
            p_gt = p_gt.double().to(device)
            ang_gt = ang_gt.double().to(device)
        
        Rot_gt = torch.zeros(Ns[1], 3, 3, device=device, dtype=dtype)
        for k in range(Ns[1]):
            ang_k = ang_gt[k]
            if device.type == 'mps':
                Rot_gt[k] = TORCHIEKF.from_rpy(ang_k[0], ang_k[1], ang_k[2]).float()
            else:
                Rot_gt[k] = TORCHIEKF.from_rpy(ang_k[0], ang_k[1], ang_k[2]).double()
        list_rpe[dataset_name] = compute_delta_p(Rot_gt[:Ns[1]], p_gt[:Ns[1]])

    list_rpe_validation = {}
    for dataset_name, Ns in dataset.datasets_validatation_filter.items():
        print(f"Processing validation dataset: {dataset_name}")
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, 0)
        
        if device.type == 'mps':
            p_gt = p_gt.float().to(device)
            ang_gt = ang_gt.float().to(device)
        else:
            p_gt = p_gt.double().to(device)
            ang_gt = ang_gt.double().to(device)
        
        Rot_gt = torch.zeros(Ns[1], 3, 3, device=device, dtype=dtype)
        for k in range(Ns[1]):
            ang_k = ang_gt[k]
            if device.type == 'mps':
                Rot_gt[k] = TORCHIEKF.from_rpy(ang_k[0], ang_k[1], ang_k[2]).float()
            else:
                Rot_gt[k] = TORCHIEKF.from_rpy(ang_k[0], ang_k[1], ang_k[2]).double()
        list_rpe_validation[dataset_name] = compute_delta_p(Rot_gt[:Ns[1]], p_gt[:Ns[1]])
    
    # ...existing code... Clean up datasets with insufficient data
    list_rpe_ = copy.deepcopy(list_rpe)
    dataset.list_rpe = {}
    for dataset_name, rpe in list_rpe_.items():
        if len(rpe[0]) is not 0:
            dataset.list_rpe[dataset_name] = list_rpe[dataset_name]
        else:
            dataset.datasets_train_filter.pop(dataset_name)
            list_rpe.pop(dataset_name)
            cprint("%s has too much dirty data, it's removed from training list" % dataset_name, 'yellow')

    list_rpe_validation_ = copy.deepcopy(list_rpe_validation)
    dataset.list_rpe_validation = {}
    for dataset_name, rpe in list_rpe_validation_.items():
        if len(rpe[0]) is not 0:
            dataset.list_rpe_validation[dataset_name] = list_rpe_validation[dataset_name]
        else:
            dataset.datasets_validatation_filter.pop(dataset_name)
            list_rpe_validation.pop(dataset_name)
            cprint("%s has too much dirty data, it's removed from validation list" % dataset_name, 'yellow')
    
    # Move data back to CPU for saving
    list_rpe_cpu = {}
    for name, rpe in list_rpe.items():
        list_rpe_cpu[name] = [rpe[0], rpe[1], rpe[2].cpu() if hasattr(rpe[2], 'cpu') else rpe[2]]
    
    list_rpe_validation_cpu = {}
    for name, rpe in list_rpe_validation.items():
        list_rpe_validation_cpu[name] = [rpe[0], rpe[1], rpe[2].cpu() if hasattr(rpe[2], 'cpu') else rpe[2]]
    
    mondict = {
        'list_rpe': list_rpe_cpu, 
        'list_rpe_validation': list_rpe_validation_cpu,
    }
    dataset.dump(mondict, file_delta_p)


def train_loop(args, dataset, epoch, iekf, optimizer, seq_dim, device):
    """Training loop with GPU acceleration"""
    loss_train = 0
    optimizer.zero_grad()
    
    for i, (dataset_name, Ns) in enumerate(dataset.datasets_train_filter.items()):
        t, ang_gt, p_gt, v_gt, u, N0 = prepare_data_filter(dataset, dataset_name, Ns, iekf, seq_dim, device)

        loss = mini_batch_step(dataset, dataset_name, iekf, dataset.list_rpe[dataset_name], t, ang_gt, p_gt, v_gt, u, N0, device)

        if loss is -1 or torch.isnan(loss):
            cprint("{} loss is invalid".format(i), 'yellow')
            continue
        elif loss > max_loss:
            cprint("{} loss is too high {:.5f}".format(i, loss), 'yellow')
            continue
        else:
            loss_train += loss
            cprint("{} loss: {:.5f}".format(i, loss))

    if loss_train == 0: 
        return 
    
    loss_train.backward()
    g_norm = torch.nn.utils.clip_grad_norm_(iekf.parameters(), max_grad_norm)
    
    # Handle gradient norm for different device types
    if device.type == 'mps':
        g_norm_val = float(g_norm)
    else:
        g_norm_val = g_norm.cpu().numpy() if hasattr(g_norm, 'cpu') else float(g_norm)
    
    if np.isnan(g_norm_val) or g_norm > 3 * max_grad_norm:
        cprint("gradient norm: {:.5f}".format(g_norm_val), 'yellow')
        optimizer.zero_grad()
    else:
        optimizer.step()
        optimizer.zero_grad()
        cprint("gradient norm: {:.5f}".format(g_norm_val))
    print('Train Epoch: {:2d} \tLoss: {:.5f}'.format(epoch, loss_train))
    return loss_train

def save_iekf(args, iekf):
    """Save IEKF model (CPU version for compatibility) - supports MPS"""
    file_name = os.path.join(args.path_temp, "iekfnets_gpu.p")
    
    # Get original device and precision info
    original_device = next(iekf.parameters()).device
    is_mps = original_device.type == 'mps'
    
    # Move to CPU for saving
    iekf_cpu = iekf.cpu()
    torch.save(iekf_cpu.state_dict(), file_name)
    
    # Move back to original device with correct precision
    if original_device.type in ['cuda', 'mps']:
        if is_mps:
            iekf = iekf.float()  # Ensure float32 for MPS
        else:
            iekf = iekf.double()  # Ensure float64 for CUDA
        iekf = iekf.to(original_device)
        
        # Handle gravity vector separately if it exists
        if hasattr(iekf, 'g') and iekf.g is not None:
            if is_mps:
                iekf.g = iekf.g.float().to(original_device)
            else:
                iekf.g = iekf.g.double().to(original_device)
    
    print("The IEKF nets are saved in the file " + file_name)

def mini_batch_step(dataset, dataset_name, iekf, list_rpe, t, ang_gt, p_gt, v_gt, u, N0, device):
    """Mini-batch step with GPU acceleration and MPS precision handling"""
    # IEKF is now device/dtype aware, no patching needed
    iekf.set_Q()
    
    measurements_covs = iekf.forward_nets(u)
    Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t, u, measurements_covs, v_gt, p_gt, t.shape[0], ang_gt[0])
    delta_p, delta_p_gt = precompute_lost(Rot, p, list_rpe, N0, device)
    if delta_p is None:
        return -1
    loss = criterion(delta_p, delta_p_gt)
    return loss

def set_optimizer(iekf):
    """Set up optimizer - same as CPU version"""
    param_list = [{
        'params': iekf.initprocesscov_net.parameters(),
        'lr': lr_initprocesscov_net,
        'weight_decay': weight_decay_initprocesscov_net
    }]
    for key, value in lr_mesnet.items():
        param_list.append({
            'params': getattr(iekf.mes_net, key).parameters(),
            'lr': value,
            'weight_decay': weight_decay_mesnet[key]
        })
    optimizer = torch.optim.Adam(param_list)
    return optimizer

def prepare_data_filter(dataset, dataset_name, Ns, iekf, seq_dim, device):
    """Prepare data for filtering with GPU support and MPS float32 compatibility"""
    # get data with trainable instant
    t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)
    t = t[Ns[0]: Ns[1]]
    ang_gt = ang_gt[Ns[0]: Ns[1]]
    p_gt = p_gt[Ns[0]: Ns[1]] - p_gt[Ns[0]]
    v_gt = v_gt[Ns[0]: Ns[1]]
    u = u[Ns[0]: Ns[1]]

    # subsample data
    N0, N = get_start_and_end(seq_dim, u)
    
    # Handle precision based on device type
    if device.type == 'mps':
        t = t[N0: N].float().to(device)
        ang_gt = ang_gt[N0: N].float().to(device)
        p_gt = (p_gt[N0: N] - p_gt[N0]).float().to(device)
        v_gt = v_gt[N0: N].float().to(device)
        u = u[N0: N].float().to(device)
    else:
        t = t[N0: N].double().to(device)
        ang_gt = ang_gt[N0: N].double().to(device)
        p_gt = (p_gt[N0: N] - p_gt[N0]).double().to(device)
        v_gt = v_gt[N0: N].double().to(device)
        u = u[N0: N].double().to(device)

    # add noise
    if iekf.mes_net.training:
        u = dataset.add_noise(u)

    return t, ang_gt, p_gt, v_gt, u, N0


def get_start_and_end(seq_dim, u):
    """Get start and end indices - same as CPU version"""
    if seq_dim is None:
        N0 = 0
        N = u.shape[0]
    else: # training sequence
        N0 = 10 * int(np.random.randint(0, (u.shape[0] - seq_dim)/10))
        N = N0 + seq_dim
    return N0, N


def precompute_lost(Rot, p, list_rpe, N0, device):
    """Precompute loss with GPU acceleration and MPS float32 compatibility"""
    N = p.shape[0]
    Rot_10_Hz = Rot[::10]
    p_10_Hz = p[::10]
    
    idxs_0 = torch.tensor(list_rpe[0], device=device, dtype=torch.long) - int(N0 / 10)
    idxs_end = torch.tensor(list_rpe[1], device=device, dtype=torch.long) - int(N0 / 10)
    delta_p_gt = list_rpe[2].to(device) if hasattr(list_rpe[2], 'to') else torch.tensor(list_rpe[2], device=device)
    
    # Ensure correct precision for MPS
    if device.type == 'mps':
        delta_p_gt = delta_p_gt.float()
    else:
        delta_p_gt = delta_p_gt.double()
    
    idxs = torch.ones(idxs_0.shape[0], device=device, dtype=torch.bool)
    idxs[idxs_0 < 0] = False
    idxs[idxs_end >= int(N / 10)] = False
    
    delta_p_gt = delta_p_gt[idxs]
    idxs_end_bis = idxs_end[idxs]
    idxs_0_bis = idxs_0[idxs]
    
    if len(idxs_0_bis) == 0: 
        return None, None     
    else:
        delta_p = Rot_10_Hz[idxs_0_bis].transpose(-1, -2).matmul(
            (p_10_Hz[idxs_end_bis] - p_10_Hz[idxs_0_bis]).unsqueeze(-1)).squeeze()
        distance = delta_p_gt.norm(dim=1).unsqueeze(-1)
        
        # Return with appropriate precision
        if device.type == 'mps':
            return delta_p.float() / distance.float(), delta_p_gt.float() / distance.float()
        else:
            return delta_p.double() / distance.double(), delta_p_gt.double() / distance.double()
