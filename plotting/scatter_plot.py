import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '../TTC_utils'))
from get_data_2D import get_data
from steptaker import steptaker
import dataloader_2D as dataloader


def scatter_plot(critic_list, steps, c_idx, fake, args):
    """plots samples of real and fake distributions for 2D example
    Inputs
    - critic_list; list of critics, trained up to entry at c_idx, inclusive
    - steps; list of steps for critics
    - c_idx; numerical identifier of critic
    - fake; initial fake data.
    - args; various arguments.

    Outputs
    - None, saves scatter in args.temp_dir/scatter/gen{}.jpg
    - Also, if frame = 00, updates args to specify view window"""

    num_samp = 1024

    target_loader = getattr(dataloader, args.target)(args.target_params, num_samp)
    target_gen = iter(target_loader)

    real = get_data(target_gen)

    if c_idx == 0:  # if you're at the start of training, specify window
        fulldata = torch.cat((real, fake), dim=0)
        mins, _ = torch.min(fulldata, dim=0)  # x min, then y min
        mins = mins.data.cpu().numpy()
        maxs, _ = torch.max(fulldata, dim=0)
        maxs = maxs.data.cpu().numpy()
        spreads = maxs - mins

        args.xwin = [mins[0] - 0.25 * spreads[0], maxs[0] + 0.25 * spreads[0]]
        args.ywin = [mins[1] - 0.25 * spreads[1], maxs[1] + 0.25 * spreads[1]]
    real = real.data.cpu().numpy()

    # APPLY TRANSFORMATIONS - if iterations >=1, apply gradient descent maps from previous critics.
    for j in range(c_idx):
        fake = steptaker(fake, critic_list[j], steps[j])

    fake = fake.data.cpu().numpy()

    plt.scatter(real[:, 0], real[:, 1], alpha=0.1, label='real')
    plt.scatter(fake[:, 0], fake[:, 1], alpha=0.1, label='fake')
    plt.xlim(args.xwin[0], args.xwin[1])
    plt.ylim(args.ywin[0], args.ywin[1])
    plt.legend(loc='upper right')
    os.makedirs(os.path.join(args.temp_dir, 'scatter'), exist_ok=True)
    plt.savefig(
        os.path.join(args.temp_dir, 'scatter',
                     'step{}.jpg'.format(c_idx)))
    plt.savefig(
        os.path.join(args.temp_dir, 'scatter',
                     'step{}.pdf'.format(c_idx)))
    plt.close()
