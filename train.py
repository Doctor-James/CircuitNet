# Copyright 2022 CircuitNet. All rights reserved.

import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from losses import build_loss
from models.build_model import build_model
from utils.configs import Paraser
from metrics import build_metric
from math import cos, pi
import sys, os, subprocess
import wandb


def checkpoint(model, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"./{save_path}/model_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
        


class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights = [1],
                 min_lr = None,
                 min_lr_ratio = None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                    end: float,
                    factor: float,
                    weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                        f'cumulative_periods {cumulative_periods}')


    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    
    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                        lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
        ]


def train():
    argp = Paraser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    if not os.path.exists(arg_dict['save_path']):
        os.makedirs(arg_dict['save_path'])
    with open(os.path.join(arg_dict['save_path'],  'arg.json'), 'wt') as f:
      json.dump(arg_dict, f, indent=4)

    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False 

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('===> Building model')
    # Initialize model parameters
    model = build_model(arg_dict)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    # model = model.cuda()
    
    # Build loss
    loss = build_loss(arg_dict)

    # Build Optimzer
    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'],  betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])

    # Build lr scheduler
    cosine_lr = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    # wandb
    os.environ["WANDB_API_KEY"] = '6c8a124fda2ff1f951266df70b7a216bb0d7f13d'
    os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="congestion_gpdl", entity="doctor-james-zjl", dir=arg_dict['save_path'])
    wandb.config = {
        "batch_size": 16,
        "max_iters": arg_dict['max_iters'],
        "learning_rate": arg_dict['lr'],
        "weight_decay": arg_dict['weight_decay'],
    }

    # metrics
    metrics = {k: build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k: 0 for k in arg_dict['eval_metric']}


    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    metrics_freq = 100
    save_freq = 10000

    while iter_num < arg_dict['max_iters']:
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in dataset:        
                input, target = feature.to(device), label.to(device)

                regular_lr = cosine_lr.get_regular_lr(iter_num)
                cosine_lr._set_lr(optimizer, regular_lr)

                prediction = model(input)

                optimizer.zero_grad()
                pixel_loss = loss(prediction, target)

                epoch_loss += pixel_loss.item()
                pixel_loss.backward()
                optimizer.step()

                iter_num += 1
                
                bar.update(1)
                wandb.log({"loss": pixel_loss})

                if iter_num % metrics_freq ==0:
                    batch_size = target.shape[0]
                    for i in range(batch_size):
                        target_ = target[i,].squeeze(1)
                        prediction_ = prediction[i,].squeeze(1)
                        for metric, metric_func in metrics.items():
                            if not metric_func(target_.cpu(), prediction_.cpu()) == 1:
                                avg_metrics[metric] += metric_func(target_.cpu(), prediction_.cpu())
                    for metric, avg_metric in avg_metrics.items():
                        # print("===> Avg. {}: {:.4f}".format(metric, avg_metric/batch_size))
                        wandb.log({'{}'.format(metric): avg_metric/batch_size})
                if iter_num % print_freq == 0:
                    break

        print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(iter_num, iter_num, arg_dict['max_iters'], epoch_loss / print_freq))
        if iter_num % save_freq == 0:
            checkpoint(model, iter_num, arg_dict['save_path'])
        epoch_loss = 0



if __name__ == "__main__":
    train()
