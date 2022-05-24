"""
Cleaned up code for running 2-D examples with Trust The Critics (TTC). Will train a list of critic networks
and visualize the critics as well as the modifications they make to the source dataset as they try to push it towards the
target.

IMPORTANT INPUTS
    - source (defaults to circle): The name of the distribution to push towards the target (see dataloader_2D.py for an explanation)
    - target (defaults to circle): The name of a distribution.
    - temp_dir (required): A directory where the figures will be saved
    - num_crit: The number of critic networks used to push the source to the target.
    - theta: The step size parameter described in the paper


OUTPUTS
Running this script will save the following files in temp_dir:
    - Scatter plots of the updated source data as well as the target
    - Contour plots of the critics at each step
    - A .pkl log file containing, among other things, the step sizes for each critic (as in eq. (14) of the paper). Saved under temp_dir/log.pkl 
    - A .txt file containing the configuration of the experiment, saved under temp_dir/train_config.txt.
"""

import os, sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'TTC_utils'))
sys.path.append(os.path.join(os.getcwd(), 'plotting'))
import argparse
import time
import log
import json
import random
import numpy as np
import torch
from torch import optim

import dataloader_2D
import networks
from get_training_time import write_training_time
from critic_trainer import critic_trainer
from contour_plot import contour_plot
from scatter_plot import scatter_plot

#################
# Get command line args
#################
parser = argparse.ArgumentParser('Training code for 2D TTC examples')
parser.add_argument('--source', type=str, required=True, default='circle',
                    choices=['circle', 'rotated_circle', 'line', 'gaussian', 'uniform', 'sin', 'spiral',
                             'hollow_rectangle'],
                    help='Which source distribution?')
parser.add_argument('--source_params', nargs='+', help='A list specifying the source dist. Enter 0 to get syntax',
                    required=True, type=float)
parser.add_argument('--target', type=str, required=True, default='circle',
                    choices=['circle', 'rotated_circle', 'line', 'gaussian', 'uniform', 'sin', 'spiral',
                             'hollow_rectangle'],
                    help='Which target distribution?')
parser.add_argument('--target_params', nargs='+', help='A list specifying the target dist. Enter 0 to get syntax',
                    required=True, type=float)
parser.add_argument('--temp_dir', type=str, required=True, help='temporary directory for saving')
parser.add_argument('--dim', type=int, default=64, help='int determining network dimensions')
parser.add_argument('--seed', type=int, default=-1, help='Set random seed for reproducibility')
parser.add_argument('--lamb', type=float, default=1000., help='parameter multiplying gradient penalty')
parser.add_argument('--clr', type=float, default=1e-4, help='learning rate for critic updates')
parser.add_argument('--theta', type=float, default=0.5,
                    help='parameter determining step size as fraction of W1 distance')
parser.add_argument('--critters', type=int, default=5, help='number of critic iters')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--plus', action='store_true', help='take one sided penalty')
parser.add_argument('--num_crit', type=int, default=5, help='number of critics to train')
parser.add_argument('--beta_1', type=float, default=0.5, help='beta_1 for Adam')
parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2 for Adam')
parser.add_argument('--relu', action='store_true', help='use relu for non-linearity. If not, use tanh')
parser.add_argument('--nice_contours', action='store_true',
                    help='use automatic spacing for contours. If False, use spacing of 1')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

print('Arguments:')
for p in vars(args).items():
    print('  ', p[0] + ': ', p[1])
print('\n')

# code to get deterministic behaviour
if args.seed != -1:  # if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # If true, optimizes convolution for hardware, but gives non-deterministic behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
else:
    torch.backends.cudnn.benchmark = True
    print('using benchmark')

# begin definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

###################
# Initialize Dataset iterators
###################

target_loader = getattr(dataloader_2D, args.target)(args.target_params, args.bs)

source_loader = getattr(dataloader_2D, args.source)(args.source_params, args.bs)
###################
# save args in config file
###################

os.makedirs(args.temp_dir, exist_ok=True)
config_file_name = os.path.join(args.temp_dir, 'train_config.txt')
with open(config_file_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

####################
# Initialize networks and optimizers
####################

critic_list = [None] * args.num_crit
steps = [1] * args.num_crit

for i in range(args.num_crit):
    critic_list[i] = getattr(networks, 'Discriminator')(args.dim, relu=args.relu)

if use_cuda:
    for i in range(args.num_crit):
        critic_list[i] = critic_list[i].cuda()

optimizer_list = [None] * args.num_crit
print('Adam parameters are {} and {}'.format(args.beta_1, args.beta_2))
for i in range(args.num_crit):
    optimizer_list[i] = optim.Adam(critic_list[i].parameters(), lr=args.clr, betas=(args.beta_1, args.beta_2))

# make plot of initial distributions
# the same initial source points are plotting at each step so one can see where individual points move
tracked_source_gen = iter(source_loader)
tracked_fake = torch.cat([next(tracked_source_gen) for b_idx in range(1024//args.bs)], dim=0)
scatter_plot(critic_list, steps, 0, tracked_fake, args)


abs_start = time.time()
# main training loop
for iteration in range(args.num_crit):
    ############################
    # (1) Train D network
    ###########################
    # trains critic at critic_list[iteration], and reports current estimate of W1 distance
    critic_list, W1_dist = critic_trainer(critic_list, optimizer_list, iteration, steps, target_loader,
                                                source_loader, args)

    ###########################
    # (2) Pick step size
    ###########################

    steps[iteration] = max(args.theta * W1_dist.detach(), 0.05*torch.ones([1]))

    ###########################
    # (3) freeze critic and save
    ###########################

    for p in critic_list[iteration].parameters():
        p.requires_grad = False  # this critic is now fixed

    if iteration < args.num_crit - 1:
        critic_list[iteration + 1].load_state_dict(
            critic_list[iteration].state_dict())  # initialize next critic at current critic

    log.plot('steps', steps[iteration].cpu().data.numpy())
    log.flush(args.temp_dir)
    log.tick()

    # ###########################
    # (4) Make contour and scatter plot
    ###########################
    scatter_plot(critic_list, steps, iteration + 1, tracked_fake, args)
    contour_plot(critic_list[iteration], iteration + 1, args)

print(steps)
write_training_time(args)
