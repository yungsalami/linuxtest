"""Unfolding Newton"""
import numpy as np
from collections import namedtuple


def C_thikonov(n_dims):

    # this is just a dummy solution for a running exercise script
    return np.ones(n_dims)


def llh_gradient(A, g, f, tau=0.0):

    # this is just a dummy solution for a running exercise script
    return np.ones(f.shape)


def llh_hessian(A, g, f, tau=0.0):

    # this is just a dummy solution for a running exercise script
    return A


def minimize(
    fun_grad, fun_hess, x0, prec=1e-10, max_iter=1000, epsilon=1e-6
):
    x = np.copy(x0)
    output = namedtuple(
        "minimization_result", ["x", "success", "hess_inv", "n_iterations"]
    )

    # remeber to set success to True if precision is reached
    success = False
    # and to set n_iterations accordinlgy
    n_iterations = max_iter

    # this is just a dummy solution for a running exercise script
    # so: overwrite H_inv
    n_dim = len(x)
    H_inv = np.zeros((n_dim, n_dim))


    return output(x=x, success=success, hess_inv=H_inv, n_iterations=n_iterations)
