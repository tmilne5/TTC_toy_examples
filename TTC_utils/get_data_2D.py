import torch


def get_data(generic_iterator):
    """Code to get minibatch from data iterator
    Inputs:
    - generic_iterator; iterator for dataset
    Outputs:
    - data; minibatch of data from iterator
    """

    data = next(generic_iterator)

    if torch.cuda.is_available():
        data = data.cuda()

    return data
