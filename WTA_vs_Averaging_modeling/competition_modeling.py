
from __future__ import division, print_function

__author__      = "Joe Donovan and Miguel Fernandes"
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec
from scipy import stats, optimize
from scipy.special import i0
from scipy.stats.stats import _validate_distribution


def vonmises_kde(data, kappa, n_bins=100):
    """
    Apply a von mises style KDE to handle circular data
    Adapted from from https://stackoverflow.com/questions/28839246/scipy-gaussian-kde-and-circular-data#
    """
    x = np.linspace(0, 2 * np.pi, n_bins)
    # integrate von mises kernels
    kde = np.exp(kappa * np.cos(x[:, None] - data[None, :])).sum(1) / (2 * np.pi * i0(kappa))
    # normalize so the integral is 1
    kde /= np.trapz(kde, x=x)
    return x, kde


def polar_average(theta1, theta2, weight=None):
    """
    Take the (weighted) vector average, since a normal mean doesn't make sense for circular quantities
    Note - theta should be in radians
    see https://en.wikipedia.org/wiki/Mean_of_circular_quantities
    """
    if weight is None:
        weight = np.ones_like(theta1) * .5
    return np.arctan2(np.sin(theta1) * weight + (1 - weight) * np.sin(theta2), np.cos(theta1) * weight + (1 - weight) * np.cos(theta2)) % (2 * np.pi)


def avg_model(s1, s2, both):
    """
    Model that chooses the average between stimuli, aka Averaging
    """
    tlen = len(both)
    return polar_average(np.random.choice(s1, tlen), np.random.choice(s2, tlen))



def comp_model(s1, s2, both):
    """
    Competition model that chooses always one stimulus or the other, aka WTA
    """
    tlen = int(len(both) / 2)
    xs1 = np.random.choice(s1, tlen)
    xs2 = np.random.choice(s2, len(both) - tlen)
    return np.hstack((xs1, xs2))


def mix_model(s1, s2, both, mix_p, tlen=None):
    """
    Simple mixture model, that blends between averaging and WTA models.
    mix_p = 0, is the WTA model
    mix_p = 1, is average model
    """
    if tlen is None:
        tlen = len(both)

    mixed = int(mix_p * tlen)
    mix_res = polar_average(np.random.choice(s1, mixed), np.random.choice(s2, mixed))

    unmixed = tlen - mixed
    return np.hstack((mix_res, np.random.choice(s1, int(unmixed / 2)), np.random.choice(s2, unmixed - int(unmixed / 2))))



def mix_model_bias(s1, s2, both, mix_p, bias, tlen=None):
    """
    Simple mixture model with a bias for s1 vs s2, that blends between avg and WTA models.
    mix_p = 0, is the WTA model
    mix_p = 1, is average model
    bias = 0, competition is always s1
    bias = 1, competition is always s2
    """
    if tlen is None:
        tlen = len(both)

    assert 0 <= bias <= 1
    assert 0 <= mix_p <= 1

    mixed = int(mix_p * tlen)
    mix_res = polar_average(np.random.choice(s1, mixed), np.random.choice(s2, mixed))

    unmixed_probs = np.hstack((np.ones(len(s1)) * (1 - bias), np.ones(len(s2)) * bias))
    unmixed_probs /= unmixed_probs.sum()
    unmixed_res = np.random.choice(np.hstack((s1, s2)), size=tlen - mixed, p=unmixed_probs)
    return np.hstack((mix_res, unmixed_res))


def tv_distance(a, b):
    return np.sum(np.abs(a - b))


def circ_distance(arr1, arr2):
    """
    Assume radians
    """
    delta = (arr1 - arr2) % (2 * np.pi)
    delta[delta > np.pi] = 2 * np.pi - delta[delta > np.pi]
    return delta


def circ_energy_distance_fast(u_values, v_values):
    """
    Compute the circular energy distance - adapted from the scipy code
    Assumes input in radians
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.energy_distance.html
    """
    u_values, u_weights = _validate_distribution(u_values, None)
    v_values, v_weights = _validate_distribution(v_values, None)

    # Modulus the input data
    u_values = u_values % (np.pi * 2)
    v_values = v_values % (np.pi * 2)

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Change deltas to circular
    dtemp = deltas % (2 * np.pi)
    dtemp[dtemp > np.pi] = 2 * np.pi - dtemp[dtemp > np.pi]
    deltas = dtemp

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    u_cdf = u_cdf_indices / u_values.size
    v_cdf = v_cdf_indices / v_values.size
    return np.sqrt(2) * np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))


def slope_rvs(b, size):
    # TODO fix b =1, shift range from 0,2 to
    r = np.random.rand(size)
    res = (-b + (4 * (1 - b) * r + b ** 2) ** .5) / (2 - 2 * b)
    r2 = np.random.randint(0, high=2, size=size)
    res = -res * (r2 * 2 - 1)
    res = (res + 1) / 2
    return res


def slope_rvs_uni(b, size):
    r = np.random.rand(size)
    res = r
    return (res / b - .5 / b) % 1


def trunc_basis_b(b=.1, size=1):
    loc = 0
    r = stats.truncnorm.rvs(-b, b, size=size, loc=loc, scale=1 / (2 * b))
    res = r % 1
    return res


def trunc_basis_sym(loc, b=16, size=1):
    r = stats.truncnorm.rvs(-b, b, size=size, loc=loc, scale=1 / (2 * b))
    res = r % 1
    r2 = np.random.randint(0, 2, size=size)
    res[r2 == 0] = 1 - res[r2 == 0]
    return res


def trunc_basis(loc, b=6, size=1):
    r = stats.truncnorm.rvs(-b, b, size=size, loc=loc, scale=1 / (2 * b))
    res = r % 1
    return res


def trunc_basis_set(nbasis, amps, size):
    amps = np.asarray(amps)
    amps = amps / amps.sum()
    locs = np.linspace(0, 1, nbasis, endpoint=False)
    ps = np.random.choice(np.arange(nbasis), size=size, p=amps)
    sizes = [np.sum(ps == i) for i in np.arange(nbasis)]
    bases = [trunc_basis(loc, b=nbasis, size=size) for loc, size in zip(locs, sizes)]
    return np.hstack(bases)


def trunc_basis_set_sym(nbasis, amps, size):
    amps = np.asarray(amps).astype(np.float)
    amps[0] = amps[0] * .5
    amps[-1] = amps[-1] * .5
    # print(amps)
    amps = amps / amps.sum()
    locs = np.linspace(0, .5, nbasis, endpoint=True)
    ps = np.random.choice(np.arange(nbasis), size=size, p=amps)
    sizes = [np.sum(ps == i) for i in np.arange(nbasis)]
    bases = [trunc_basis_sym(loc, b=nbasis * 1.7, size=size) for loc, size in zip(locs, sizes)]
    return np.hstack(bases)


if __name__ == '__main__':
    # Plots have been moved to the comp_modeling_plots.ipybnb
    # TODO anyway to get error bars for dists? not so easy...

    # TODO could also add some kind of neighboorhood mixing
    pass
