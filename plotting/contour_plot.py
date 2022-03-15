import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def contour_plot(critic, iteration, args):
    """
    Generates a contour plot of the critic
    """
    num_points = 100

    points = np.zeros((num_points, num_points, 2), dtype='float32')
    points[:, :, 0] = np.linspace(args.xwin[0], args.xwin[1], num_points)[:, None]
    points[:, :, 1] = np.linspace(args.ywin[0], args.ywin[1], num_points)[None, :]
    points = points.reshape((-1, 2))
    points = torch.from_numpy(points)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        points = torch.Tensor(points).cuda()

    critic_vals = critic(points)  # applies discriminator to points.
    critic_vals = critic_vals.data.cpu().numpy()
    if args.nice_contours:
        levels = 100  # automatically spaces 100 contour lines for visual appeal
    else:
        levels = np.arange(np.amin(critic_vals), np.amax(critic_vals))  # there will be spacing of 1 between contours

    x = np.linspace(args.xwin[0], args.xwin[1], num_points)
    y = np.linspace(args.ywin[0], args.ywin[1], num_points)
    plt.contour(x, y, critic_vals.reshape((len(x), len(y))).transpose(),
                levels=levels)  # creates contour map of critic
    plt.title('Critic Contour Plot')
    plt.colorbar()
    os.makedirs(os.path.join(args.temp_dir, 'contours'), exist_ok=True)
    plt.savefig(os.path.join(args.temp_dir, 'contours',
                             'critters{}_step{}_dim{}_lamb{}.pdf'.format(args.critters, iteration,
                                                                                  args.dim, args.lamb
                                                                                  )), dpi=300)

    plt.close()
