# New BSD License
#
# Copyright (c) 2007 - 2012 The scikit-learn developers.
# All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the Name of the Scikit-learn Developers  nor the Names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
'''
Utility functions taken from scikit-learn
'''

import inspect
import itertools

import numpy as np
from scipy import linalg
import math
import pickle

def quaternion_to_euler(x, y, z, w):
    x, y, z, w = float(x), float(y), float(z), float(w)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [roll, pitch, yaw]

def array1d(X, dtype=None, order=None):
    """Returns at least 1-d array with data from X"""
    return np.asarray(np.atleast_1d(X), dtype=dtype, order=order)


def array2d(X, dtype=None, order=None):
    """Returns at least 2-d array with data from X"""
    return np.asarray(np.atleast_2d(X), dtype=dtype, order=order)


def log_multivariate_normal_density(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices. """
    if hasattr(linalg, 'solve_triangular'):
        # only in scipy since 0.9
        solve_triangular = linalg.solve_triangular
    else:
        # slower, but works
        solve_triangular = linalg.solve
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probabily stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                      lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) + \
                                     n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{0} cannot be used to seed a numpy.random.RandomState'
                     + ' instance').format(seed)


class Bunch(dict):
    """Container object for datasets: dictionary-like object that exposes its
    keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def get_params(obj):
    '''Get Names and values of all parameters in `obj`'s __init__'''
    try:
        # get Names of every variable in the argument
        args = inspect.getargspec(obj.__init__)[0]
        args.pop(0)   # remove "self"

        # get values for each of the above in the object
        argdict = dict([(arg, obj.__getattribute__(arg)) for arg in args])
        return argdict
    except:
        raise ValueError("object has no __init__ method")


def preprocess_arguments(argsets, converters):
    """convert and collect arguments in order of priority

    Parameters
    ----------
    argsets : [{argName: argval}]
        a list of argument sets, each with lower levels of priority
    converters : {argName: function}
        conversion functions for each argument

    Returns
    -------
    result : {argName: argval}
        processed arguments
    """
    result = {}
    for argset in argsets:
        for (argName, argval) in argset.items():
            # check that this argument is necessary
            if not argName in converters:
                raise ValueError("Unrecognized argument: {0}".format(argName))

            # potentially use this argument
            if argName not in result and argval is not None:
                # convert to right type
                argval = converters[argName](argval)

                # save
                result[argName] = argval

    # check that all arguments are covered
    if not len(converters.keys()) == len(result.keys()):
        missing = set(converters.keys()) - set(result.keys())
        s = "The following arguments are missing: {0}".format(list(missing))
        raise ValueError(s)

    return result

import matplotlib.pyplot as plt
import io
import PIL
from PIL import Image
from torchvision.transforms import ToTensor
from random import randint

def plot_tensorboard(writer,Datas, Lines, Labels,Name="Image",ylabel=None,ylim=None):
    rnd = randint(0,10000)
    with open(Name+".pkl", "wb") as open_file:
        pickle.dump(Datas, open_file)

    if Name == "Image":
        Name = Name + str(rnd)
    plt.figure(rnd,figsize=(14,7))
        ## code to plot the image in tensorboard
    plt.title(Name)
    if ylabel is not None:
         plt.ylabel(ylabel)
    plt.xlabel("Time [0.01s]")
    if ylim is not None:
        axes = plt.gca()
        axes.set_ylim(ylim)
    times = [i for i in range(len(Datas[0]))]
    for data,line,label in zip(Datas, Lines, Labels):
        plt.plot(times, data, line,label=label)
        plt.legend(loc="lower right")

    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    writer.add_image(Name, image, 0)
    plt.clf()
    plt.cla()
    plt.close()

def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")

