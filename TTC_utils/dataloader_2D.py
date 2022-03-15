import torch
import numpy as np


def param_splitter(params):
    """A simple function that takes a string of params separated by underscores and returns them as floats
    Inputs
    - params; a string of the form param1_param2_param3...
    Outputs
    - list_params; a list of floats"""
    list_params = params.split('_')
    list_params = [float(item) for item in list_params]

    return list_params


def circle(params, bs, track_modes=False):
    """A generator which samples several Gaussians, distributed at equally spaced points on the unit circle
    Inputs
    - std_x; standard deviation of the Gaussians in x direction (in params)
    - std_y; standard deviation of the Gaussians in y direction (in params)
    - num_gauss; how many Gaussians to place on the circle (in params)
    - offset; horizontal center of circle (in params)
    - bs; batch size
    - theta; angle of rotation
    - track_modes; if true, returns data sorted by the mode it came from. In this case, num_gauss should divide args.bs
    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = CircleGaussianLoader()

    loader.bs = bs
    p_list = param_splitter(params)
    loader.std_x, loader.std_y, loader.num_gauss, loader.offset, loader.theta = p_list[0], p_list[1], p_list[2], p_list[
        3], p_list[4]
    loader.num_gauss = int(loader.num_gauss)
    loader.track_modes = track_modes

    return loader


class CircleGaussianLoader:

    def __iter__(self):
        return self

    def __next__(self):
        x = np.matmul(np.random.randn(self.bs, 2), np.array([[self.std_x, 0], [0, self.std_y]]))
        angles = np.linspace(0, 2 * 3.14159, self.num_gauss, endpoint=False)
        biases_bank = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
        if self.track_modes:
            indices = np.arange(0, self.num_gauss)
            indices = np.repeat(indices, self.bs // self.num_gauss)
        else:
            indices = np.random.randint(0, self.num_gauss, self.bs)
        biases = biases_bank[indices, :]
        rotation = np.array([[np.cos(self.theta), np.sin(self.theta)], [-np.sin(self.theta), np.cos(self.theta)]])
        x = np.matmul(x + biases, rotation)
        x = x + np.array([self.offset, 0])

        x = torch.Tensor(x)

        return x


def rotated_circle(params, bs, track_modes=False):
    """A generator which samples several Gaussians, distributed at equally spaced points on the unit circle. Differs from
    circle in that no biases are used; instead rotations are applied.
    Inputs
    - std_x; standard deviation of the Gaussians in x direction (in params)
    - std_y; standard deviation of the Gaussians in y direction (in params)
    - num_gauss; how many Gaussians to place on the circle (in params)
    - offset; horizontal center of circle (in params)
    - theta; angle of initial rotation
    - bs; batch size
    - track_modes; if true, returns data sorted by the mode it came from. In this case, num_gauss should divide args.bs
    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = RotatedCircleGaussianLoader()

    loader.bs = bs
    p_list = param_splitter(params)
    loader.std_x, loader.std_y, loader.num_gauss, loader.offset, loader.theta = p_list[0], p_list[1], p_list[2], p_list[
        3], p_list[4]
    loader.num_gauss = int(loader.num_gauss)
    loader.track_modes = track_modes

    return loader


class RotatedCircleGaussianLoader:

    def __iter__(self):
        return self

    def __next__(self):
        x = np.matmul(np.random.randn(self.bs, 2), np.array([[self.std_x, 0], [0, self.std_y]])) + np.array([1, 0])
        angles = np.linspace(self.theta, self.theta + 2 * 3.14159, self.num_gauss, endpoint=False)
        rotations_bank = np.array(
            [[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]] for angle in angles])
        if self.track_modes:
            indices = np.arange(0, self.num_gauss)
            indices = np.repeat(indices, self.bs // self.num_gauss)
        else:
            indices = np.random.randint(0, self.num_gauss, self.bs)
        rotations = rotations_bank[indices, :, :]
        x_list = np.stack([np.matmul(x[i, :], rotations[i, :, :]) for i in range(self.bs)], axis=0)
        x = x_list
        x = x + np.array([self.offset, 0])

        x = torch.Tensor(x)

        return x


def line(params, bs, track_modes=False):
    """A generator which samples several Gaussians, distributed at equally spaced points on an interval
    Inputs
    - std; standard deviation of the Gaussians (contained in args)
    - num_gauss; how many Gaussians to place on the line (contained in args)
    - offset; horizontal bias of Gaussians (contained in args)
    - vspace; vertical span of gaussians (contained in args)
    - bs; minibatch size
    - track_modes; if true, returns data sorted by the mode it came from. In this case, num_gauss should divide args.bs
    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = LineGaussianLoader()

    loader.bs = bs  # num_gauss should divide this.
    p_list = param_splitter(params)
    loader.std, loader.num_gauss, loader.offset, loader.vspace = p_list[0], p_list[1], p_list[2], p_list[3]
    loader.num_gauss = int(loader.num_gauss)
    loader.track_modes = track_modes

    return loader


class LineGaussianLoader:

    def __iter__(self):
        return self

    def __next__(self):
        x = self.std * np.random.randn(self.bs, 2)
        vertical_offsets = np.linspace(-self.vspace, self.vspace, self.num_gauss)
        biases_bank = np.array([[self.offset, vertical_offset] for vertical_offset in vertical_offsets])
        if self.track_modes:
            indices = np.arange(0, self.num_gauss)
            indices = np.repeat(indices, self.bs // self.num_gauss)
        else:
            indices = np.random.randint(0, self.num_gauss, self.bs)

        biases = biases_bank[indices, :]

        x = x + biases

        x = torch.Tensor(x)

        return x


def gaussian(params, bs, track_modes=False):
    """A generator which samples a Gaussian with selectable mean as well as x and y variance
    Inputs
    - params; string in the form shift_dimension_stdx_stdy
    - bs; minibatch size
    - latent_dim; dimension of space where Gaussian will be generated
    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = GaussianLoader()

    loader.bs = bs  # num_gauss should divide this.
    p_list = param_splitter(params)
    loader.offset, loader.latent_dim, loader.stdx, loader.stdy = p_list[0], p_list[1], p_list[2], p_list[3]
    loader.latent_dim = int(loader.latent_dim)

    return loader


class GaussianLoader:

    def __iter__(self):
        return self

    def __next__(self):
        if self.latent_dim == 2:

            bias = np.array([[self.offset, 0]])
            bias = np.concatenate([bias for i in range(self.bs)], axis=0)  # repeats bias several times
            scale = np.array([[self.stdx, 0], [0, self.stdy]])
            bias = torch.Tensor(bias)
            scale = torch.Tensor(scale)

            output = torch.matmul(torch.randn([self.bs, self.latent_dim]), scale) + bias
        else:
            output = torch.randn([self.bs, self.latent_dim])

        return output


def sin(params, bs, track_modes=False):
    """A generator which samples the graph of a sin wave with selectable frequency, horizontal/vertical shift and horizontal span
    Inputs
    - params; string in the form frequency_hshift_vshift_hspan_amp.
        - hshift; subtracted from input before multiplication by frequency.
        - frequency; determines the frequency of the wave. Multiplied by 2pi, and multiplies shifted input
        - amp; amplitude of the wave
        - vshift; added to output.
        - hspan; the total length of the interval over which to sample the wave.


    - bs; minibatch size
    -
    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = SinLoader()

    loader.bs = bs  # num_gauss should divide this.
    p_list = param_splitter(params)
    loader.hshift, loader.freq, loader.amp, loader.vshift, loader.hspan = p_list[0], p_list[1], p_list[2], p_list[3], \
                                                                          p_list[4]

    return loader


class SinLoader:

    def __iter__(self):
        return self

    def __next__(self):
        xvals = (np.random.rand(self.bs) - 1 / 2) * self.hspan
        yvals = np.sin(self.freq * 2 * 3.14159 * (xvals - self.hshift))
        yvals = self.amp * yvals + self.vshift
        outputs = np.stack((xvals, yvals), axis=1)

        outputs = torch.Tensor(outputs)
        return outputs


def spiral(params, bs, track_modes=False):
    """A generator which samples the polar curve r = a theta
    Inputs
    - params; string in the form a_numrot.
        - a; float determining the frequency of the spiral
        - numrot; number of rotations

    - bs; minibatch size

    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = SpiralLoader()

    loader.bs = bs  # num_gauss should divide this.
    p_list = param_splitter(params)
    loader.a, loader.numrot = p_list[0], p_list[1]

    return loader


class SpiralLoader:

    def __iter__(self):
        return self

    def __next__(self):
        thetas = self.numrot * 2 * 3.14159 * np.random.rand(self.bs)
        xvals = self.a * thetas * np.cos(thetas)
        yvals = self.a * thetas * np.sin(thetas)
        outputs = np.stack((xvals, yvals), axis=1)

        outputs = torch.Tensor(outputs)
        return outputs


def uniform(params, bs, track_modes=False):
    """A generator which samples a uniform distribution over a specified rectangle
    Inputs
    - params; string in the form xlow_xhigh_ylow_yhigh, which specifies the corners of the rectangle
    - bs; minibatch size
    -
    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = UniformLoader()

    loader.bs = bs  # num_gauss should divide this.
    p_list = param_splitter(params)
    loader.xlow, loader.xhigh, loader.ylow, loader.yhigh = p_list[0], p_list[1], p_list[2], p_list[3]

    return loader


class UniformLoader:

    def __iter__(self):
        return self

    def __next__(self):
        xvals = np.random.rand(self.bs) * (self.xhigh - self.xlow) + self.xlow
        yvals = np.random.rand(self.bs) * (self.yhigh - self.ylow) + self.ylow
        outputs = np.stack((xvals, yvals), axis=1)
        outputs = torch.Tensor(outputs)
        return outputs


def hollowrectangle(params, bs, track_modes=False):
    """A generator which samples a uniform distribution over the boundaries of a specified rectangle
    Inputs
    - params; string in the form xlow_xhigh_ylow_yhigh, which specifies the corners of the rectangle
    - bs; minibatch size
    -
    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = HollowRectangleLoader()

    loader.bs = bs
    p_list = param_splitter(params)
    loader.xlow, loader.xhigh, loader.ylow, loader.yhigh = p_list[0], p_list[1], p_list[2], p_list[3]

    return loader


class HollowRectangleLoader:

    def __iter__(self):
        return self

    def __next__(self):
        scaler = lambda a, b, c: a * (b-c) + c

        vals = np.random.rand(self.bs)
        indices = np.random.randint(0, 4, self.bs)  # choices of side
        left = np.stack((self.xlow * np.ones_like(vals[indices == 0]), scaler(vals[indices == 0], self.ylow, self.yhigh)), axis=1)
        up = np.stack((scaler(vals[indices == 1], self.xlow, self.xhigh), self.yhigh * np.ones_like(vals[indices == 1])), axis=1)
        right = np.stack((self.xhigh * np.ones_like(vals[indices == 2]), scaler(vals[indices == 2], self.ylow, self.yhigh)), axis=1)
        down = np.stack((scaler(vals[indices == 3], self.xlow, self.xhigh), self.ylow * np.ones_like(vals[indices == 3])), axis=1)

        outputs = np.concatenate((left, up, right, down), axis=0)
        outputs = outputs[np.random.permutation(self.bs), :]  # randomizes edges
        outputs = torch.Tensor(outputs)
        return outputs


if __name__ == "__main__":
    parameters = '0.1_10_1_0.5'
    circle_loader = circle(parameters, 128)
    line_loader = line(parameters, 128)
    gaussian_loader = gaussian(parameters, 128)
