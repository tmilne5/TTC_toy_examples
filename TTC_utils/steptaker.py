import torch
from torch import autograd


def steptaker(data, critic, step, num_step=1):
    """Applies gradient descent (GD) to data using critic
    Inputs
    - data; data to apply GD to
    - critic; critic to compute gradients of
    - step; how large of a step to take
    - num_step; how finely to discretize flow. taken as 1 in TTC
    Outputs
    - data with gradient descent applied
    """

    for j in range(num_step):
        gradients = grad_calc(data, critic)

        data = (data - (step / num_step) * gradients).detach()

    return data.detach()


def grad_calc(data, critic):
    """Returns the gradients of critic at data"""
    data = data.detach().clone()
    data.requires_grad = True
    Dfake = critic(data)

    gradients = autograd.grad(outputs=Dfake, inputs=data,
                              grad_outputs=torch.ones(Dfake.size()).cuda() if torch.cuda.is_available() else torch.ones(
                                  Dfake.size()), only_inputs=True)[0]
    return gradients.detach()
