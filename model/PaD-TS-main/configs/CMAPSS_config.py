import argparse
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.abspath(os.path.join(current_dir, '../../')))

class Training_args(argparse.Namespace):
    data_dir = ""
    schedule_sampler = "batch"
    lr = 1e-4
    weight_decay = 0.0
    lr_anneal_steps = 100000
    microbatch = -1  # -1 disables microbatches
    log_interval = 10
    save_interval = 5e3
    mmd_alpha = 0.0005
    save_dir = "./output/C-MAPSS/"


class Model_args(argparse.Namespace):
    hidden_size = 256
    num_heads = 4
    n_encoder = 1
    n_decoder = 3
    feature_last = True
    mlp_ratio = 4.0
    input_shape = (33, 14)


class Diffusion_args(argparse.Namespace):
    predict_xstart = True
    diffusion_steps = 500
    noise_schedule = "cosine"
    loss = "MSE_MMD"
    rescale_timesteps = False


class DataLoader_args(argparse.Namespace):
    batch_size = 64
    shuffle = True
    num_workers = 0
    drop_last = True
    pin_memory = True


class Data_args(argparse.Namespace):
    name = "FD001"
    proportion = 1.0
    data_root = os.path.join(root_dir,"dataProcess","C-MAPSS","output")
    window = 24
    save2npy = True
    neg_one_to_one = True
    seed = 123
    period = "train"
    dim = 14
