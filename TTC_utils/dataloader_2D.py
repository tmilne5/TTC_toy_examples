import torch
import numpy as np
import sys


def circle(params, bs):
    """A generator which samples several Gaussians, distributed at equally spaced points on the unit circle
    Inputs
    - std_x; standard deviation of the Gaussians in x direction (in params)
    - std_y; standard deviation of the Gaussians in y direction (in params)
    - num_gauss; how many Gaussians to place on the circle (in params)
    - offset; horizontal center of circle (in params)
    - theta; angle of rotation
    - bs; batch size
    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = CircleGaussianLoader()

    loader.bs = bs
    try:
        loader.std_x, loader.std_y, loader.num_gauss, loader.offset, loader.theta = params[0], params[1], params[2], \
                                                                                    params[3], params[4]
    except IndexError:
        print('Samples several gaussians distributed on the unit circle. Does not rotate covariance matrices')
        print('Syntax: std_x, std_y, num_gauss, horizontal offset, angle of initial rotation')
        sys.exit()

    loader.num_gauss = int(loader.num_gauss)

    return loader


class CircleGaussianLoader:

    def __iter__(self):
        return self

    def __next__(self):
        # generate Gaussian data with diagonal covariance matrix
        x = np.matmul(np.random.randn(self.bs, 2), np.array([[self.std_x, 0], [0, self.std_y]]))
        angles = np.linspace(0, 2 * 3.14159, self.num_gauss, endpoint=False)  # angles of means of Gaussians
        biases_bank = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])  # means of Gaussians

        indices = np.random.randint(0, self.num_gauss, self.bs)
        biases = biases_bank[indices, :]  # makes random selections of biases
        rotation = np.array([[np.cos(self.theta), np.sin(self.theta)], [-np.sin(self.theta), np.cos(self.theta)]])
        x = np.matmul(x + biases, rotation)  # applies rotation matrix to full dataset
        x = x + np.array([self.offset, 0])  # adds horizontal offset

        x = torch.Tensor(x)

        return x


def rotated_circle(params, bs):
    """A generator which samples several Gaussians, distributed at equally spaced points on the unit circle. Differs from
    circle in that no biases are used; instead rotations are applied.
    Inputs
    - std_x; standard deviation of the Gaussians in x direction (in params)
    - std_y; standard deviation of the Gaussians in y direction (in params)
    - num_gauss; how many Gaussians to place on the circle (in params)
    - offset; horizontal center of circle (in params)
    - theta; angle of initial rotation
    - bs; batch size
    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = RotatedCircleGaussianLoader()

    loader.bs = bs

    try:
        loader.std_x, loader.std_y, loader.num_gauss, loader.offset, loader.theta = params[0], params[1], params[2], \
                                                                                    params[3], params[4]
    except IndexError:
        print('Samples gaussians distributed at equally spaced points on the unit circle. Rotation is applied to covariance')
        print('Syntax: std_x, std_y, num_gauss, horizontal offset, angle of initial rotation')
        sys.exit()

    loader.num_gauss = int(loader.num_gauss)

    return loader


class RotatedCircleGaussianLoader:

    def __iter__(self):
        return self

    def __next__(self):
        # generate Gaussian data with diagonal covariance
        x = np.matmul(np.random.randn(self.bs, 2), np.array([[self.std_x, 0], [0, self.std_y]])) + np.array([1, 0])

        # generate list of rotations
        angles = np.linspace(self.theta, self.theta + 2 * 3.14159, self.num_gauss, endpoint=False)
        rotations_bank = np.array(
            [[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]] for angle in angles])

        # randomly assign rotations
        indices = np.random.randint(0, self.num_gauss, self.bs)
        rotations = rotations_bank[indices, :, :]

        # apply rotations
        x_list = np.stack([np.matmul(x[i, :], rotations[i, :, :]) for i in range(self.bs)], axis=0)
        # add offset
        x = x_list + np.array([self.offset, 0])

        x = torch.Tensor(x)

        return x


def line(params, bs):
    """A generator which samples several Gaussians, distributed at equally spaced points on an interval
    Inputs
    - std; standard deviation of the Gaussians (contained in args)
    - num_gauss; how many Gaussians to place on the line (contained in args)
    - offset; horizontal bias of Gaussians (contained in args)
    - vspace; vertical span of gaussians (contained in args)
    - bs; minibatch size
    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = LineGaussianLoader()

    loader.bs = bs  # num_gauss should divide this.

    try:
        loader.std, loader.num_gauss, loader.offset, loader.vspace = params[0], params[1], params[2], params[3]
    except IndexError:
        print('Samples several gaussians with diagonal covariance on a specified line')
        print('Syntax: std, number of gaussians, horizontal offset, total vertical span of line')
        sys.exit()

    loader.num_gauss = int(loader.num_gauss)

    return loader


class LineGaussianLoader:

    def __iter__(self):
        return self

    def __next__(self):
        # Generate IID gaussian noise
        x = self.std * np.random.randn(self.bs, 2)
        # List of means for each Gaussian
        vertical_offsets = np.linspace(-self.vspace, self.vspace, self.num_gauss)
        biases_bank = np.array([[self.offset, vertical_offset] for vertical_offset in vertical_offsets])
        # Randomly choose biases
        indices = np.random.randint(0, self.num_gauss, self.bs)
        biases = biases_bank[indices, :]

        x = x + biases

        x = torch.Tensor(x)

        return x


def gaussian(params, bs):
    """A generator which samples a Gaussian with selectable mean as well as x and y variance
    Inputs
    - params; string in the form shift_dimension_stdx_stdy
    - bs; minibatch size
    - latent_dim; dimension of space where Gaussian will be generated
    Outputs
    - Dataloader which samples the desired distribution
    """
    loader = GaussianLoader()

    loader.bs = bs

    try:
        loader.offset, loader.stdx, loader.stdy = params[0], params[1], params[2]
    except IndexError:
        print('Samples a Gaussian with centre on the diagonal y = x, with specified std in either dimension')
        print('Syntax: offset (where each component will be equal), std in x, std in y ')
        sys.exit()

    return loader


class GaussianLoader:

    def __iter__(self):
        return self

    def __next__(self):
        # Applies affine transformation to gaussian noise
        bias = self.offset * 2 ** (-1 / 2) * np.ones([1, 2])
        bias = np.concatenate([bias for i in range(self.bs)], axis=0)
        bias = torch.Tensor(bias)

        scale = np.array([[self.stdx, 0], [0, self.stdy]])
        scale = torch.Tensor(scale)

        output = torch.matmul(torch.randn([self.bs, 2]), scale) + bias
        return output


def sin(params, bs):
    """A generator which samples the graph of a sin wave with selectable frequency, horizontal/vertical shift and horizontal span
    Inputs
    - params; string in the form frequency_hshift_vshift_hspan_amp.
        - hshift; subtracted from input before multiplication by frequency.
        - frequency determines the frequency of the wave. Multiplied by 2pi, and multiplies shifted input
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

    try:
        loader.hshift, loader.freq, loader.amp, loader.vshift, loader.hspan = params[0], params[1], params[2], \
                                                                              params[3], params[4]
    except IndexError:
        print('Samples sin wave of the form y = a sin(f(x-b)) + c over centered interval')
        print('Syntax: horizontal shift, frequency, amplitude, vertical shift, horizontal span ')
        sys.exit()

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


def spiral(params, bs):
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

    loader.bs = bs

    try:
        loader.a, loader.numrot = params[0], params[1]
    except IndexError:
        print('Samples polar curve r = a theta uniformly')
        print('Syntax: frequency of spiral (a), number of rotations.  ')
        sys.exit()

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


def uniform(params, bs):
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

    try:
        loader.xlow, loader.xhigh, loader.ylow, loader.yhigh = params[0], params[1], params[2], params[3]
    except IndexError:
        print('Samples uniform distribution over specified rectangle')
        print('Syntax: xlow, xhigh, ylow, yhigh')
        sys.exit()

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


def hollow_rectangle(params, bs):
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

    try:
        loader.xlow, loader.xhigh, loader.ylow, loader.yhigh = params[0], params[1], params[2], params[3]
    except IndexError:
        print('Samples a uniform distribution over the boundary of a specified rectangle')
        print('Syntax: xlow, xhigh, ylow, yhigh')
        sys.exit()

    return loader


class HollowRectangleLoader:

    def __iter__(self):
        return self

    def __next__(self):
        scaler = lambda a, b, c: a * (b - c) + c

        vals = np.random.rand(self.bs)
        indices = np.random.randint(0, 4, self.bs)  # choices of side
        # for left edge, x's are fixed at xlow, and y's are random. And so on...
        left = np.stack(
            (self.xlow * np.ones_like(vals[indices == 0]), scaler(vals[indices == 0], self.ylow, self.yhigh)), axis=1)
        up = np.stack(
            (scaler(vals[indices == 1], self.xlow, self.xhigh), self.yhigh * np.ones_like(vals[indices == 1])), axis=1)
        right = np.stack(
            (self.xhigh * np.ones_like(vals[indices == 2]), scaler(vals[indices == 2], self.ylow, self.yhigh)), axis=1)
        down = np.stack(
            (scaler(vals[indices == 3], self.xlow, self.xhigh), self.ylow * np.ones_like(vals[indices == 3])), axis=1)

        outputs = np.concatenate((left, up, right, down), axis=0)
        outputs = outputs[np.random.permutation(self.bs), :]  # randomizes edges
        outputs = torch.Tensor(outputs)
        return outputs


if __name__ == "__main__":
    parameters = '0.1_10_1_0.5'
    circle_loader = circle(parameters, 128)
    line_loader = line(parameters, 128)
    gaussian_loader = gaussian(parameters, 128)

