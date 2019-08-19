# -*- coding: utf-8 -*-

# Authors: Tal Linzen <linzen@nyu.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD (3-clause)

from inspect import isgenerator
from collections import namedtuple

import numpy as np
from scipy import linalg, sparse


def LinearModel(data, design_matrix, names):
    """Aux function."""
    from scipy import stats
    n_samples = len(data)
    n_features = np.product(data.shape[1:])
    if design_matrix.ndim != 2:
        raise ValueError('Design matrix must be a 2d array')
    n_rows, n_predictors = design_matrix.shape

    if n_samples != n_rows:
        raise ValueError('Number of rows in design matrix must be equal '
                         'to number of observations')
    if n_predictors != len(names):
        raise ValueError('Number of regressor names must be equal to '
                         'number of column in design matrix')

    y = np.reshape(data, (n_samples, n_features))
    betas, resid_sum_squares, _, _ = linalg.lstsq(a=design_matrix, b=y)
    df = n_rows - n_predictors
    sqrt_noise_var = np.sqrt(resid_sum_squares / df).reshape(data.shape[1:])
    design_invcov = linalg.inv(np.dot(design_matrix.T, design_matrix))
    unscaled_stderrs = np.sqrt(np.diag(design_invcov))
    tiny = np.finfo(np.float64).tiny
    beta, stderr, t_val, p_val, mlog10_p_val = (dict() for _ in range(5))
    for x, unscaled_stderr, predictor in zip(betas, unscaled_stderrs, names):
        beta[predictor] = x.reshape(data.shape[1:])
        stderr[predictor] = sqrt_noise_var * unscaled_stderr
        p_val[predictor] = np.empty_like(stderr[predictor])
        t_val[predictor] = np.empty_like(stderr[predictor])

        stderr_pos = (stderr[predictor] > 0)
        beta_pos = (beta[predictor] > 0)
        t_val[predictor][stderr_pos] = (beta[predictor][stderr_pos] /
                                        stderr[predictor][stderr_pos])
        cdf = stats.t.cdf(np.abs(t_val[predictor][stderr_pos]), df)
        p_val[predictor][stderr_pos] = np.clip((1. - cdf) * 2., tiny, 1.)
        # degenerate cases
        mask = (~stderr_pos & beta_pos)
        t_val[predictor][mask] = np.inf * np.sign(beta[predictor][mask])
        p_val[predictor][mask] = tiny
        # could do NaN here, but hopefully this is safe enough
        mask = (~stderr_pos & ~beta_pos)
        t_val[predictor][mask] = 0
        p_val[predictor][mask] = 1.
        mlog10_p_val[predictor] = -np.log10(p_val[predictor])

    return beta, stderr, t_val, p_val, mlog10_p_val


