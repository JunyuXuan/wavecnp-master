import abc
import bisect
import warnings
import numpy as np
import time

import lab as B
B.epsilon = 1e-6
import stheno.torch
import torch
from sklearn.model_selection import train_test_split

from pytorch_wavelets import DWTForward, DWTInverse

from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D

from wavecnp.utils import device

__all__ = ['GPGenerator', 'SawtoothGenerator']

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

DATASETS_DICT = {
    "mnist": "MNIST",
    "svhn": "SVHN",
    "celeba32": "CelebA32",
    "celeba64": "CelebA64",
    "zs-multi-mnist": "ZeroShotMultiMNIST",
    "zsmm": "ZeroShotMultiMNIST",  # shorthand
    "zsmmt": "ZeroShotMultiMNISTtrnslt",
    "zsmms": "ZeroShotMultiMNISTscale",
    "zs-mnist": "ZeroShotMNIST",
    "celeba": "CelebA",
    "celeba128": "CelebA128",
}
DATASETS = list(DATASETS_DICT.keys())


def _rand(val_range, *shape):
    lower, upper = val_range
    return lower + np.random.rand(*shape) * (upper - lower)

def _rand_torch(val_range, *shape):
    lower, upper = val_range
    return lower + torch.rand(*shape) * (upper - lower)


def _uprank(a):
    if len(a.shape) == 1:
        return a[:, None, None]
    elif len(a.shape) == 2:
        return a[:, :, None]
    elif len(a.shape) == 3:
        return a
    else:
        return ValueError(f'Incorrect rank {len(a.shape)}.')


class LambdaIterator:
    """Iterator that repeatedly generates elements from a lambda.

    Args:
        generator (function): Function that generates an element.
        num_elements (int): Number of elements to generate.
    """

    def __init__(self, generator, num_elements):
        self.generator = generator
        self.num_elements = num_elements
        self.index = 0

    def __next__(self):
        self.index += 1
        if self.index <= self.num_elements:
            return self.generator()
        else:
            raise StopIteration()

    def __iter__(self):
        return self


class DataGenerator(metaclass=abc.ABCMeta):
    """Data generator for GP samples.

    Args:
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch.
            Defaults to 256.
        x_range (tuple[float], optional): Range of the inputs. Defaults to
            [-2, 2].
        max_train_points (int, optional): Number of training points. Must be at
            least 3. Defaults to 50.
        max_test_points (int, optional): Number of testing points. Must be at
            least 3. Defaults to 50.
    """

    def __init__(self,
                 batch_size=16,
                 num_tasks=256,
                 x_range=(-2, 2),
                 max_train_points=50,
                 max_test_points=50):
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
        self.max_train_points = max(max_train_points, 3)
        self.max_test_points = max(max_test_points, 3)

    @abc.abstractmethod
    def sample(self, x):
        """Sample at inputs `x`.

        Args:
            x (vector): Inputs to sample at.

        Returns:
            vector: Sample at inputs `x`.
        """

    def generate_task(self):
        """Generate a task.

        Returns:
            dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
                `y_context`, `x_target`, and `y_target.
        """
        tic1 = time.perf_counter()

        task = {'x': [],
                'y': [],
                'x_context': [],
                'y_context': [],
                'x_target': [],
                'y_target': []}

        # Determine number of test and train points.
        num_train_points = np.random.randint(3, self.max_train_points + 1)
        num_test_points = np.random.randint(3, self.max_test_points + 1)
        num_points = num_train_points + num_test_points

        for i in range(self.batch_size):
            # Sample inputs and outputs.
            x = _rand(self.x_range, num_points)
            y = self.sample(x)

            # Determine indices for train and test set.
            inds = np.random.permutation(x.shape[0])
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            # Record to task.
            task['x'].append(sorted(x))
            task['y'].append(y[np.argsort(x)])
            task['x_context'].append(x[inds_train])
            task['y_context'].append(y[inds_train])
            task['x_target'].append(x[inds_test])
            task['y_target'].append(y[inds_test])

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(_uprank(np.stack(v, axis=0)),
                                dtype=torch.float32).to(device)
                for k, v in task.items()}

        toc1 = time.perf_counter()
        print(" ---- stack batch uses time: ", toc1 - tic1)

        return task

    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.num_tasks)


class DataGenerator_torch(metaclass=abc.ABCMeta):
    """Data generator for GP samples.

    Args:
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch.
            Defaults to 256.
        x_range (tuple[float], optional): Range of the inputs. Defaults to
            [-2, 2].
        max_train_points (int, optional): Number of training points. Must be at
            least 3. Defaults to 50.
        max_test_points (int, optional): Number of testing points. Must be at
            least 3. Defaults to 50.
    """

    def __init__(self,
                 batch_size=16,
                 num_tasks=256,
                 x_range=(-2, 2),
                 max_train_points=50,
                 max_test_points=50):
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
        self.max_train_points = max(max_train_points, 3)
        self.max_test_points = max(max_test_points, 3)

    @abc.abstractmethod
    def sample(self, x):
        """Sample at inputs `x`.

        Args:
            x (vector): Inputs to sample at.

        Returns:
            vector: Sample at inputs `x`.
        """

    def generate_task(self):
        """Generate a task.

        Returns:
            dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
                `y_context`, `x_target`, and `y_target.
        """
        # tic1 = time.perf_counter()

        # Determine number of test and train points.
        # num_train_points_np = np.random.randint(3, self.max_train_points + 1)
        # num_test_points_np = np.random.randint(3, self.max_test_points + 1)
        # num_points = num_train_points + num_test_points
        num_train_points = int(np.power(2, np.floor(np.real(np.log2(np.random.randint(3, self.max_train_points + 1))))))
        num_test_points = int(np.power(2, np.floor(np.real(np.log2(np.random.randint(3, self.max_test_points + 1))))))
        num_points = num_train_points + num_test_points

        task = {'x': torch.zeros(self.batch_size, num_points, 1, dtype=torch.float32).to(device),
                'y': torch.zeros(self.batch_size, num_points, 1, dtype=torch.float32).to(device),
                'x_context': torch.zeros(self.batch_size, num_train_points, 1, dtype=torch.float32).to(device),
                'y_context': torch.zeros(self.batch_size, num_train_points, 1, dtype=torch.float32).to(device),
                'x_target': torch.zeros(self.batch_size, num_test_points, 1, dtype=torch.float32).to(device),
                'y_target': torch.zeros(self.batch_size, num_test_points, 1, dtype=torch.float32).to(device)
                }

        inds_train = torch.zeros(num_points)
        inds_train[:num_train_points] = 1

        for i in range(self.batch_size):
            # Sample inputs and outputs.
            x = _rand_torch(self.x_range, num_points)
            y = self.sample(x)

            # Record to task.
            x_sorted, ind_sorted = torch.sort(x)

            ind_perm = torch.randperm(num_points)
            ind_trains = inds_train[ind_perm].bool()
            ind_target = (1-inds_train[ind_perm]).bool()

            task['x'][i, :, 0] = x_sorted
            task['y'][i, :, 0] = y[ind_sorted]
            task['x_context'][i, :, 0] = x[ind_trains]
            task['y_context'][i, :, 0] = y[ind_trains]
            task['x_target'][i, :, 0] = x[ind_target]
            task['y_target'][i, :, 0] = y[ind_target]

        # toc1 = time.perf_counter()
        # print(" ---- stack batch uses time: ", toc1 - tic1)

        return task

    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.num_tasks)


class GPGenerator_torch(DataGenerator_torch):
    """Generate samples from a GP with a given kernel.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel=stheno.torch.EQ(), **kw_args):
        self.gp = stheno.torch.GP(kernel)
        DataGenerator_torch.__init__(self, **kw_args)

    def sample(self, x):
        # return np.squeeze(self.gp(x).sample())
        return self.gp(x).sample().squeeze()


class GPGenerator(DataGenerator):
    """Generate samples from a GP with a given kernel.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel=stheno.EQ(), **kw_args):
        self.gp = stheno.GP(kernel)
        DataGenerator_torch.__init__(self, **kw_args)

    def sample(self, x):
        return np.squeeze(self.gp(x).sample())
        # return self.gp(x).sample().squeeze()


class DataGenerator_task(metaclass=abc.ABCMeta):
    """
    """

    def __init__(self,
                 batch_size=16,
                 num_tasks=256,
                 x_range=(-2, 2),
                 max_train_points=50,
                 max_test_points=50):
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
        self.max_train_points = max(max_train_points, 3)
        self.max_test_points = max(max_test_points, 3)

    @abc.abstractmethod
    def sample(self, x):
        """Sample at inputs `x`.

        Args:
            x (vector): Inputs to sample at.

        Returns:
            vector: Sample at inputs `x`.
        """

    def generate_task(self):
        """Generate a task.

        Returns:
            dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
                `y_context`, `x_target`, and `y_target.
        """
        task = {'x': [],
                'y0': [],
                'y1': [],
                'y2': [],
                'y3': [],
                'x_context': [],
                'y_context': [],
                'x_target': [],
                'y_target': [],
                'ind': []}

        # Determine number of test and train points.
        num_train_points = int(np.power(2, np.floor(np.real(np.log2(np.random.randint(3, self.max_train_points + 1))))))
        num_test_points = int(np.power(2, np.floor(np.real(np.log2(np.random.randint(3, self.max_test_points + 1))))))
        num_points = num_train_points + num_test_points

        for i in range(self.batch_size):
            # Sample inputs and outputs.
            x = _rand(self.x_range, num_points)
            y, y_all, ind = self.sample(x)

            # Determine indices for train and test set.
            inds = np.random.permutation(x.shape[0])
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            # Record to task.
            task['x'].append(sorted(x))
            task['y0'].append(np.squeeze(y_all[0, np.argsort(x)]))
            task['y1'].append(np.squeeze(y_all[1, np.argsort(x)]))
            task['y2'].append(np.squeeze(y_all[2, np.argsort(x)]))
            task['y3'].append(np.squeeze(y_all[3, np.argsort(x)]))
            task['x_context'].append(x[inds_train])
            task['y_context'].append(y[inds_train])
            task['x_target'].append(x[inds_test])
            task['y_target'].append(y[inds_test])
            task['ind'].append(ind)

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(_uprank(np.stack(v, axis=0)),
                                dtype=torch.float32).to(device)
                for k, v in task.items()}

        return task

    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.num_tasks)


class GPGenerator_task(DataGenerator_task):
    """Generate samples from a GP with different resolutions.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel=stheno.EQ(), **kw_args):
        self.gp = stheno.GP(kernel)
        DataGenerator_task.__init__(self, **kw_args)

        # parameters of dwt
        self.level = 3  # int(log(num_points) / log(2.))
        self.wavename = 'db2'
        self.L = 4  # filter coefficient vecotr len
        self.mode = 'zero'

        self.dwt = DWT1DForward(wave=self.wavename, J=self.level).to(device)
        self.iwt = DWT1DInverse(wave=self.wavename).to(device)

    def sample(self, x):

        # Generate a base function
        h = torch.tensor(self.gp(x).sample(), dtype=torch.float32).to(device)
        h = h[:, :, None].permute(2, 1, 0)
        x_num = x.size

        # Apply dwt
        yl, yh = self.dwt(h)
        # tmph = self.iwt((yl, yh))

        # Output functions with different resolutions
        h_all = torch.zeros(self.level + 1, x_num)
        h_all[0] = h.squeeze()

        for l in range(self.level):
            tmpyh = torch.zeros_like(yh[l]) + 0.0001
            yh[l] = tmpyh
            tmp = self.iwt((yl, yh))

            if tmp.size(2) != x_num:
                print("EORRY in sample()!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            h_all[l + 1] = tmp.squeeze()

        # Randomly select a resolution for all point
        ind = torch.randint(0, self.level + 1, (1,))
        # rnd_mask = torch.nn.functional.one_hot(ind, num_classes=self.level+1)

        h_all_ind = h_all[ind, :]
        h_all_ind = np.squeeze(h_all_ind.detach().numpy())
        h_all = h_all.detach().numpy()
        ind = ind.detach().numpy()

        return h_all_ind, h_all, ind


class GPGenerator_torch_task(DataGenerator_torch):
    """Generate samples from a GP with different resolutions.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel=stheno.torch.EQ(), **kw_args):
        self.gp = stheno.torch.GP(kernel)
        DataGenerator_torch.__init__(self, **kw_args)

        # parameters of dwt
        self.level = 3  # int(log(num_points) / log(2.))
        self.wavename = 'db2'
        self.L = 4  # filter coefficient vecotr len
        self.mode = 'zero'

        self.dwt = DWT1DForward(wave=self.wavename, J=self.level).to(device)
        self.iwt = DWT1DInverse(wave=self.wavename).to(device)

    def sample(self, x):

        # Generate a base function
        h = torch.tensor(self.gp(x).sample(), dtype=torch.float32).to(device)
        h = h[:, :, None].permute(2, 1, 0)
        x_num = x.shape[0]

        # Apply dwt
        yl, yh = self.dwt(h)
        # tmph = self.iwt((yl, yh))

        # Output functions with different resolutions
        h_all = torch.zeros(self.level + 1, x_num).to(device)
        h_all[0] = h.squeeze()

        for l in range(self.level):
            tmpyh = torch.zeros_like(yh[l]).to(device) + 0.0001
            yh[l] = tmpyh
            tmp = self.iwt((yl, yh))

            if tmp.size(2) != x_num:
                print("EORRY in sample()!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            h_all[l + 1] = tmp.squeeze()

        # Randomly select a resolution for all point
        ind = torch.randint(0, self.level + 1, (1,)).to(device)
        # rnd_mask = torch.nn.functional.one_hot(ind, num_classes=self.level+1)

        h_all_ind = h_all[ind, :]
        h_all_ind = h_all_ind.squeeze()
        # h_all = h_all.detach().numpy()
        # ind = ind.detach().numpy()

        return h_all_ind


class DataGenerator_sample(metaclass=abc.ABCMeta):
    """
    """

    def __init__(self,
                 batch_size=16,
                 num_tasks=256,
                 x_range=(-2, 2),
                 max_train_points=50,
                 max_test_points=50):
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
        self.max_train_points = max(max_train_points, 3)
        self.max_test_points = max(max_test_points, 3)

    @abc.abstractmethod
    def sample(self, x):
        """Sample at inputs `x`.

        Args:
            x (vector): Inputs to sample at.

        Returns:
            vector: Sample at inputs `x`.
        """

    def generate_task(self):
        """Generate a task.

        Returns:
            dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
                `y_context`, `x_target`, and `y_target.
        """

        task = {'x': [],
                'y0': [],
                'y1': [],
                'y2': [],
                'y3': [],
                'y': [],
                'x_context': [],
                'y_context': [],
                'x_target': [],
                'y_target': [],
                'ind': []}

        # Determine number of test and train points.
        num_train_points = int(np.power(2, np.floor(np.real(np.log2(np.random.randint(3, self.max_train_points + 1))))))
        num_test_points = int(np.power(2, np.floor(np.real(np.log2(np.random.randint(3, self.max_test_points + 1))))))
        num_points = num_train_points + num_test_points

        for i in range(self.batch_size):
            # Sample inputs and outputs.
            x = _rand(self.x_range, num_points)
            y, y_all, ind = self.sample(x)

            # Determine indices for train and test set.
            inds = np.random.permutation(x.shape[0])
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            # Record to task.
            task['x'].append(sorted(x))
            task['y0'].append(np.squeeze(y_all[0, np.argsort(x)]))
            task['y1'].append(np.squeeze(y_all[1, np.argsort(x)]))
            task['y2'].append(np.squeeze(y_all[2, np.argsort(x)]))
            task['y3'].append(np.squeeze(y_all[3, np.argsort(x)]))
            task['y'].append(y[np.argsort(x)])
            task['x_context'].append(x[inds_train])
            task['y_context'].append(y[inds_train])
            task['x_target'].append(x[inds_test])
            task['y_target'].append(y[inds_test])
            task['ind'].append(ind[np.argsort(x)])

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(_uprank(np.stack(v, axis=0)),
                                dtype=torch.float32).to(device)
                for k, v in task.items()}

        return task

    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.num_tasks)


class GPGenerator_sample(DataGenerator_sample):
    """Generate samples from a GP with different resolutions.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel=stheno.EQ(), **kw_args):
        self.gp = stheno.GP(kernel)
        DataGenerator_sample.__init__(self, **kw_args)

        # parameters of dwt
        self.level = 3  # int(log(num_points) / log(2.))
        self.wavename = 'db2'
        self.L = 4  # filter coefficient vecotr len
        self.mode = 'zero'

        self.dwt = DWT1DForward(wave=self.wavename, J=self.level)
        self.iwt = DWT1DInverse(wave=self.wavename)

    def sample(self, x):

        # Generate a base function
        h = torch.tensor(self.gp(x).sample(), dtype=torch.float32)
        h = h[:, :, None].permute(2, 1, 0)
        x_num = x.size

        # Apply dwt
        yl, yh = self.dwt(h)

        # Output functions with different resolutions
        h_all = torch.zeros(self.level + 1, x_num)
        h_all[0] = h.squeeze()

        for l in range(self.level):
            tmpyh = torch.zeros_like(yh[l]) + 0.0001
            # tmpyh.to(device)
            yh[l] = tmpyh
            tmp = self.iwt((yl, yh))

            if tmp.size(2) != x_num:
                print("ERORR in sample() !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            h_all[l + 1] = tmp.squeeze()

        # Randomly select resolution for each point
        # TODO: consider keep half of orignal function
        inds = torch.randint(0, self.level + 1, (x_num,))
        rnd_mask = torch.nn.functional.one_hot(inds, num_classes=self.level + 1)

        h_all_ind = h_all * torch.transpose(rnd_mask, 1, 0)
        h_all_ind = np.squeeze(h_all_ind.sum(0).detach().numpy())

        return h_all_ind, h_all, inds


class GPGenerator_torch_sample(DataGenerator_torch):
    """Generate samples from a GP with different resolutions.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel=stheno.EQ(), **kw_args):
        self.gp = stheno.GP(kernel)
        DataGenerator_torch.__init__(self, **kw_args)

        # parameters of dwt
        self.level = 3  # int(log(num_points) / log(2.))
        self.wavename = 'db2'
        self.L = 4  # filter coefficient vecotr len
        self.mode = 'zero'

        self.dwt = DWT1DForward(wave=self.wavename, J=self.level).to(device)
        self.iwt = DWT1DInverse(wave=self.wavename).to(device)

    def sample(self, x):

        # Generate a base function
        h = torch.tensor(self.gp(x).sample(), dtype=torch.float32).to(device)
        h = h[:, :, None].permute(2, 1, 0)
        x_num = x.shape[0]

        # Apply dwt
        yl, yh = self.dwt(h)

        # Output functions with different resolutions
        h_all = torch.zeros(self.level + 1, x_num).to(device)
        h_all[0] = h.squeeze()

        for l in range(self.level):
            tmpyh = torch.zeros_like(yh[l]).to(device) + 0.0001
            # tmpyh.to(device)
            yh[l] = tmpyh
            tmp = self.iwt((yl, yh))

            if tmp.size(2) != x_num:
                print("ERORR in sample() !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            h_all[l + 1] = tmp.squeeze()

        # Randomly select resolution for each point
        # TODO: consider keep half of orignal function
        inds = torch.randint(0, self.level + 1, (x_num,)).to(device)
        rnd_mask = torch.nn.functional.one_hot(inds, num_classes=self.level + 1).to(device)

        h_all_ind = h_all * torch.transpose(rnd_mask, 1, 0)
        h_all_ind = h_all_ind.sum(0).squeeze()

        return h_all_ind


# class GPGenerator_sample(DataGenerator):
#     """Generate samples from a GP with different resolutions.
#
#     Further takes in keyword arguments for :class:`.data.DataGenerator`.
#
#     Args:
#         kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
#             Defaults to an EQ kernel.
#     """
#
#     def __init__(self, kernel=stheno.EQ(), **kw_args):
#         self.gp = stheno.GP(kernel)
#         DataGenerator.__init__(self, **kw_args)
#
#         # parameters of dwt
#         self.level = 3  # int(log(num_points) / log(2.))
#         self.wavename = 'db2'
#         self.L = 4  # filter coefficient vecotr len
#         self.mode = 'zero'
#
#         self.dwt = DWT1DForward(wave=self.wavename, J=self.level)
#         self.iwt = DWT1DInverse(wave=self.wavename)
#
#     def sample(self, x):
#
#         FINISH = False
#
#         while not FINISH:
#             # Generate a base function
#             h = torch.tensor(self.gp(x).sample(), dtype=torch.float32).to(device)
#             h = h[:,:,None].permute(2, 1, 0)
#             x_num = x.size
#
#             # Apply dwt
#             yl, yh = self.dwt(h)
#
#             # Output functions with different resolutions
#             h_all = torch.zeros(self.level+1, x_num)
#             h_all[0] = h.squeeze()
#
#             for l in range(self.level):
#                 tmpyh = torch.zeros_like(yh[l]) + 0.0001
#                 yh[l] = tmpyh
#                 tmp = self.iwt((yl, yh))
#
#                 if tmp.size(2) != x_num:
#                     FINISH = False
#                     continue
#
#                 h_all[l+1] = tmp.squeeze()
#
#             # Randomly select resolution for each point
#             # TODO: consider keep half of orignal function
#             inds = torch.randint(0, self.level, (x_num,))
#             rnd_mask = torch.nn.functional.one_hot(inds, num_classes=self.level+1)
#
#             h_all = h_all * torch.transpose(rnd_mask, 1, 0)
#             h_all = np.squeeze(h_all.sum(0).detach().numpy())
#
#             break
#
#         return h_all


#
# class GPGenerator_task(DataGenerator):
#     """Generate samples from a GP with different resolutions.
#
#     Further takes in keyword arguments for :class:`.data.DataGenerator`.
#
#     Args:
#         kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
#             Defaults to an EQ kernel.
#     """
#
#     def __init__(self, kernel=stheno.EQ(), **kw_args):
#         self.gp = stheno.GP(kernel)
#         DataGenerator.__init__(self, **kw_args)
#
#         # parameters of dwt
#         self.level = 3  # int(log(num_points) / log(2.))
#         self.wavename = 'db2'
#         self.L = 4  # filter coefficient vecotr len
#         self.mode = 'zero'
#
#         self.dwt = DWT1DForward(wave=self.wavename, J=self.level).to(device)
#         self.iwt = DWT1DInverse(wave=self.wavename).to(device)
#
#     def sample(self, x):
#
#         FINISH = False
#
#         while not FINISH:
#             # Generate a base function
#             h = torch.tensor(self.gp(x).sample(), dtype=torch.float32).to(device)
#             h = h[:,:,None].permute(2, 1, 0)
#             x_num = x.size
#
#             # Apply dwt
#             yl, yh = self.dwt(h)
#
#             # Output functions with different resolutions
#             h_all = torch.zeros(self.level+1, x_num)
#             h_all[0] = h.squeeze()
#
#             for l in range(self.level):
#                 tmpyh = torch.zeros_like(yh[l]) + 0.0001
#                 yh[l] = tmpyh
#                 tmp = self.iwt((yl, yh))
#
#                 if tmp.size(2) != x_num:
#                     FINISH = False
#                     continue
#
#                 h_all[l+1] = tmp.squeeze()
#
#             # Randomly select a resolution for all point
#             ind = torch.randint(0, self.level, (1,))
#             # rnd_mask = torch.nn.functional.one_hot(ind, num_classes=self.level+1)
#
#             h_all_ind = h_all[ind, :]
#             h_all_ind = np.squeeze(h_all_ind.detach().numpy())
#             h_all = np.squeeze(h_all.detach().numpy())
#
#             break
#
#         return h_all_ind


class SawtoothGenerator(DataGenerator):
    """Generate samples from a random sawtooth.

    Further takes in keyword arguments for :class:`.data.DataGenerator`. The
    default numbers for `max_train_points` and `max_test_points` are 100.

    Args:
        freq_dist (tuple[float], optional): Lower and upper bound for the
            random frequency. Defaults to [3, 5].
        shift_dist (tuple[float], optional): Lower and upper bound for the
            random shift. Defaults to [-5, 5].
        trunc_dist (tuple[float], optional): Lower and upper bound for the
            random truncation. Defaults to [10, 20].
    """

    def __init__(self,
                 freq_dist=(3, 5),
                 shift_dist=(-5, 5),
                 trunc_dist=(10, 20),
                 max_train_points=100,
                 max_test_points=100,
                 **kw_args):
        self.freq_dist = freq_dist
        self.shift_dist = shift_dist
        self.trunc_dist = trunc_dist
        DataGenerator.__init__(self,
                               max_train_points=max_train_points,
                               max_test_points=max_test_points,
                               **kw_args)

    def sample(self, x):
        # Sample parameters of sawtooth.
        amp = 1
        freq = _rand(self.freq_dist)
        shift = _rand(self.shift_dist)
        trunc = np.random.randint(self.trunc_dist[0], self.trunc_dist[1] + 1)

        # Construct expansion.
        x = x[:, None] + shift
        k = np.arange(1, trunc + 1)[None, :]
        return 0.5 * amp - amp / np.pi * \
               np.sum((-1) ** k * np.sin(2 * np.pi * k * freq * x) / k, axis=1)



# 2d DATA

def convert_grid_2_real(x_context, image_size_1, image_size_2):

    x_context[:, :, 0] *= 2 / (image_size_1 - 1)  # in [0,2]
    x_context[:, :, 0] -= 1  # in [-1,1]

    x_context[:, :, 1] *= 2 / (image_size_2 - 1)  # in [0,2]
    x_context[:, :, 1] -= 1  # in [-1,1]

    return x_context


def convert_ind_2_task(cnt_mask_ind):
    batch_size = cnt_mask_ind.shape[0]
    image_size_1 = cnt_mask_ind.shape[-2]
    image_size_2 = cnt_mask_ind.shape[-1]

    cnt_mask_task = torch.zeros(batch_size, image_size_1, image_size_2)
    cnt_mask_task.to(device)

    for b in range(batch_size):
        cnt_mask_task[b, :, :] = cnt_mask_ind[b, :, :, :].sum(0)

    return cnt_mask_task

def generate_rand_2d_mask_ind(batch_size, image_size_1, image_size_2, cnt_num):
    num_pixels = image_size_1 * image_size_2
    idx = torch.rand(batch_size, num_pixels, device=device).topk(cnt_num, dim=1).indices.sort(dim=1).values
    cnt_mask_ind = torch.zeros(batch_size, cnt_num, num_pixels, device=device)
    cnt_mask_ind.scatter_(2, idx.unsqueeze(-1), 1)
    return cnt_mask_ind.view(batch_size, cnt_num, image_size_1, image_size_2)

def generate_rand_2d_mask_cnp(imgs, batch_size, image_size_1, image_size_2, cnt_num):
    num_pixels = image_size_1 * image_size_2
    idx = torch.rand(batch_size, num_pixels, device=imgs.device).topk(cnt_num, dim=1).indices.sort(dim=1).values
    cnt_mask_ind = torch.zeros(batch_size, cnt_num, num_pixels, device=imgs.device)
    cnt_mask_ind.scatter_(2, idx.unsqueeze(-1), 1)
    cnt_mask_cnp = torch.stack((idx // image_size_2, idx % image_size_2), dim=-1)
    cnt_y_cnp = imgs[:, 0].reshape(batch_size, num_pixels).gather(1, idx).unsqueeze(-1)
    return cnt_mask_ind.view(batch_size, cnt_num, image_size_1, image_size_2), cnt_mask_cnp, cnt_y_cnp

def preprocss_img_resolution_sample(imgs, batch_size, image_size_1, image_size_2):

    y_dim = imgs.shape[1]

    # parameters of dwt
    level = 3  # int(log(num_points) / log(2.))
    wavename = 'db2'
    L = 4  # filter coefficient vecotr len
    mode = 'zero'

    dwt = DWTForward(J=level, wave=wavename, mode=mode)
    dwt.to(device)
    iwt = DWTInverse(wave=wavename, mode=mode)
    iwt.to(device)

    # print("imgs - ", imgs.device)
    Yl, Yh = dwt(imgs)

    imgs_new = torch.zeros(batch_size, level+1, image_size_1, image_size_2).to(device)
    imgs_new[:, 0, :, :] = imgs.squeeze()

    for l in range(level):
        tmpyh = torch.zeros_like(Yh[l]).to(device)
        Yh[l] = tmpyh
        tmp = iwt((Yl, Yh)).squeeze()
        imgs_new[:, l+1, :, :] = tmp

    indices = torch.randint(0, level + 1, (batch_size, image_size_1, image_size_2), device=imgs.device)
    selected = torch.gather(imgs_new, 1, indices.unsqueeze(1)).squeeze(1)
    imgs.copy_(selected.unsqueeze(1).expand(-1, y_dim, -1, -1))

    return imgs


def preprocss_img_resolution_task(imgs, batch_size, image_size_1, image_size_2):
    y_dim = imgs.shape[1]

    # parameters of dwt
    level = 3  # int(log(num_points) / log(2.))
    wavename = 'db2'
    L = 4  # filter coefficient vecotr len
    mode = 'zero'

    dwt = DWTForward(J=level, wave=wavename, mode=mode).to(device)
    iwt = DWTInverse(wave=wavename, mode=mode).to(device)

    Yl, Yh = dwt(imgs)

    imgs_new = torch.zeros(batch_size, level + 1, image_size_1, image_size_2).to(device)
    imgs_new[:, 0, :, :] = imgs.squeeze()

    for l in range(level):
        tmpyh = torch.zeros_like(Yh[l]).to(device)
        Yh[l] = tmpyh
        tmp = iwt((Yl, Yh)).squeeze()
        imgs_new[:, l + 1, :, :] = tmp

    inds = torch.randint(0, level + 1, (batch_size,), device=imgs.device)
    selected = imgs_new[torch.arange(batch_size, device=imgs.device), inds]
    imgs.copy_(selected.unsqueeze(1).expand(-1, y_dim, -1, -1))

    return imgs


def get_dataset(dataset):
    """Return the correct uninstantiated datasets."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_datasets(datasets):
    """Return the correct instantiated train and test datasets."""
    train_datasets, test_datasets = dict(), dict()
    for d in datasets:
        train_datasets[d], test_datasets[d] = get_train_test_img_dataset(d)

    return train_datasets, test_datasets


def get_train_test_img_dataset(dataset):
    """Return the correct instantiated train and test datasets."""
    try:
        train_dataset = get_dataset(dataset)(split="train")
        test_dataset = get_dataset(dataset)(split="test")
    except TypeError as e:
        train_dataset, test_dataset = train_dev_split(
            get_dataset(dataset)(), dev_size=0.1, is_stratify=False
        )

    return train_dataset, test_dataset


def train_dev_split(to_split, dev_size=0.1, seed=123, is_stratify=True):
    """Split a training dataset into a training and validation one.

    Parameters
    ----------
    dev_size: float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the dev split. If int, represents the absolute
        number of dev samples.

    seed: int
        Random seed.

    is_stratify: bool
        Whether to stratify splits based on class label.
    """
    n_all = len(to_split)
    idcs_all = list(range(n_all))
    stratify = to_split.targets if is_stratify else None
    idcs_train, indcs_val = train_test_split(
        idcs_all, stratify=stratify, test_size=dev_size, random_state=seed
    )
    train = _DatasetSubset(to_split, idcs_train)
    valid = _DatasetSubset(to_split, indcs_val)

    return train, valid


class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py


class IterableDataset(Dataset[T_co]):
    r"""An iterable Dataset.

    All datasets that represent an iterable of data samples should subclass it.
    Such form of datasets is particularly useful when data come from a stream.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this dataset.

    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the dataset will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When :attr:`num_workers > 0`, each worker process will have a
    different copy of the dataset object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
    :attr:`worker_init_fn` option to modify each copy's behavior.

    Example 1: splitting workload across all workers in :meth:`__iter__`::

        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         worker_info = torch.utils.data.get_worker_info()
        ...         if worker_info is None:  # single-process data loading, return the full iterator
        ...             iter_start = self.start
        ...             iter_end = self.end
        ...         else:  # in a worker process
        ...             # split workload
        ...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        ...             worker_id = worker_info.id
        ...             iter_start = self.start + worker_id * per_worker
        ...             iter_end = min(iter_start + per_worker, self.end)
        ...         return iter(range(iter_start, iter_end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [3, 4, 5, 6]

        >>> # Mult-process loading with two worker processes
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [3, 5, 4, 6]

        >>> # With even more workers
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20)))
        [3, 4, 5, 6]

    Example 2: splitting workload across all workers using :attr:`worker_init_fn`::

        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [3, 4, 5, 6]
        >>>
        >>> # Directly doing multi-process loading yields duplicate data
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [3, 3, 4, 4, 5, 5, 6, 6]

        >>> # Define a `worker_init_fn` that configures each dataset copy differently
        >>> def worker_init_fn(worker_id):
        ...     worker_info = torch.utils.data.get_worker_info()
        ...     dataset = worker_info.dataset  # the dataset copy in this worker process
        ...     overall_start = dataset.start
        ...     overall_end = dataset.end
        ...     # configure the dataset to only process the split workload
        ...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        ...     worker_id = worker_info.id
        ...     dataset.start = overall_start + worker_id * per_worker
        ...     dataset.end = min(dataset.start + per_worker, overall_end)
        ...

        >>> # Mult-process loading with the custom `worker_init_fn`
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
        [3, 5, 4, 6]

        >>> # With even more workers
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
        [3, 4, 5, 6]
    """
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def __add__(self, other: Dataset[T_co]):
        return ChainDataset([self, other])

    # No `def __len__(self)` default? Subclasses raise `TypeError` when needed.
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]


class ChainDataset(IterableDataset):
    r"""Dataset for chaining multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ChainDataset, self).__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)  # type: ignore[arg-type]
        return total


class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class _DatasetSubset(Dataset):
    """Helper to split train dataset into train and dev dataset.

    Parameters
    ----------
    to_split: Dataset
        Dataset to subset.

    idx_mapping: array-like
        Indices of the subset.

    Notes
    -----
    - Modified from: https: // gist.github.com / Fuchai / 12f2321e6c8fa53058f5eb23aeddb6ab
    - Cannot modify the length and targets with indexing anymore! I.e.
    `d.targets[1]=-1` doesn't work because np.array doesn't allow `arr[i][j]=-1`
    but you can do `d.targets=targets`
    """

    def __init__(self, to_split, idx_mapping):
        self.idx_mapping = idx_mapping
        self.length = len(idx_mapping)
        self.to_split = to_split

    def __getitem__(self, index):
        return self.to_split[self.idx_mapping[index]]

    def __len__(self):
        return self.length

    @property
    def targets(self):
        return self.to_split.targets[self.idx_mapping]

    @targets.setter
    def targets(self, values):
        self.to_split.targets[self.idx_mapping] = values

    @property
    def data(self):
        return self.to_split.data[self.idx_mapping]

    def __getattr__(self, attr):
        return getattr(self.to_split, attr)
