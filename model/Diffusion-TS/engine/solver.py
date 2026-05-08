import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join("..", os.path.dirname(current_dir))
sys.path.append(root_dir)

from Utils.io_utils import instantiate_from_config, get_model_parameters_info


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()

        self.model = model
        self.device = self.model.betas.device

        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']

        self.dl = cycle(dataloader['dataloader'])
        self.dataloader = dataloader['dataloader']

        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger

        # ===============================
        # checkpoint dir
        # ===============================
        self.results_folder = Path(
            os.path.join(root_dir, config['solver']['results_folder'] + f'_{model.seq_length}')
        )
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # ===============================
        # optimizer & EMA
        # ===============================
        start_lr = config['solver'].get('base_lr', 1.0e-4)

        self.opt = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=start_lr,
            betas=[0.9, 0.96]
        )

        self.ema = EMA(
            self.model,
            beta=config['solver']['ema']['decay'],
            update_every=config['solver']['ema']['update_interval']
        ).to(self.device)

        # ===============================
        # scheduler
        # ===============================
        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        self.log_frequency = 100

    # =====================================================
    # NEW: batch sanitization (关键修改点)
    # =====================================================
    def _sanitize_batch(self, batch):
        """
        保证进入模型的数据：
        - torch.Tensor
        - float32
        - 不包含非数值维度
        """
        if isinstance(batch, torch.Tensor):
            x = batch
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch['data']
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # 强制 float
        if not torch.is_floating_point(x):
            x = x.float()

        return x.to(self.device)

    # =====================================================
    # Save / Load
    # =====================================================
    def save(self, milestone):
        save_path = self.results_folder / f'checkpoint-{milestone}.pt'
        torch.save({
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }, save_path)

    def load(self, milestone):
        load_path = self.results_folder / f'checkpoint-{milestone}.pt'
        data = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.step = data['step']
        self.milestone = milestone

    # =====================================================
    # Train
    # =====================================================
    def train(self):
        step = 0

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    raw_batch = next(self.dl)
                    data = self._sanitize_batch(raw_batch)

                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()

                self.step += 1
                step += 1
                self.ema.update()

                if self.step % self.save_cycle == 0:
                    self.milestone += 1
                    self.save(self.milestone)

                pbar.set_description(f'loss: {total_loss:.6f}')
                pbar.update(1)

        print("training complete")

    # =====================================================
    # Sample
    # =====================================================
    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        samples = np.empty([0, shape[0], shape[1]])

        # num_cycle = int(num // size_every) + 1
        num_cycle = 1
        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(
                batch_size=size_every,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn
            )
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        return samples

    # =====================================================
    # Restore / Infill
    # =====================================================
    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        samples, reals, masks = [], [], []

        for x, t_m in raw_dataloader:
            x = self._sanitize_batch(x)
            t_m = t_m.to(self.device)

            sample = self.ema.ema_model.fast_sample_infill(
                shape=x.shape,
                target=x * t_m,
                partial_mask=t_m,
                model_kwargs={
                    'coef': coef,
                    'learning_rate': stepsize
                },
                sampling_timesteps=sampling_steps
            )

            samples.append(sample.cpu().numpy())
            reals.append(x.cpu().numpy())
            masks.append(t_m.cpu().numpy())

        return (
            np.concatenate(samples),
            np.concatenate(reals),
            np.concatenate(masks)
        )
