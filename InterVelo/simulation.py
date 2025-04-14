import numpy as np
from anndata import AnnData
import random

def unspliced(s, s_, gamma, beta):
    """TODO."""
    u=(gamma * s + s_)/beta
    return u

# TODO: Add docstrings
def spliced(tau, a, h, t_):
    """TODO."""
    s = h * np.exp(-a * (tau-t_)**2)
    s_ = -2 * a * (tau-t_) * h * np.exp(-a * (tau-t_)**2)
    return s, s_


def simulation(
    n_obs=300,
    n_vars=None,
    noise_model="normal",
    noise_level=1,
    random_seed=0,
):
    """Simulation of mRNA  kinetics.

    Simulated mRNA metabolism with radial basis function.
    The parameters for each reaction are randomly sampled from a log-normal distribution or uniform distribution,
    and time events follow the Poisson law. 

    Returns
    -------
    Returns `adata` object
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    def draw_poisson(n):
        from random import seed, uniform  # draw from poisson

        seed(random_seed)
        t = np.cumsum([-0.1 * np.log(uniform(0, 1)) for _ in range(n - 1)])
        return np.insert(t, 0, 0)  # prepend t0=0

    def simulate_dynamics(tau, a, h, t_, beta, gamma, noise_model, noise_level):
        st, st_ = spliced(tau, a, h, t_)
        ut = unspliced(st, st_, gamma, beta)
        ut, st = np.clip(ut, 0, None), np.clip(st, 0, None)
        if noise_model == "normal":  # add noise
            ut += np.random.normal(
                scale=noise_level * np.percentile(ut, 99) / 10, size=len(ut)
            )
            st += np.random.normal(
                scale=noise_level * np.percentile(st, 99) / 10, size=len(st)
            )
           
        ut, st = np.clip(ut, 0, None), np.clip(st, 0, None)
        return ut, st, st_

    t = draw_poisson(n_obs)
    t = t/np.max(t)

    # switching time point obtained as fraction of t_max rounded down
    t_ = np.random.uniform(-1, 2, n_vars)
    a = np.random.lognormal(mean=-2, sigma=0.5, size=n_vars)
    h = np.random.lognormal(mean=2, sigma=1, size=n_vars)
    beta = np.random.lognormal(mean=0, sigma=0.3, size=n_vars)
    gamma = np.random.lognormal(mean=0, sigma=0.3, size=n_vars)

    U = np.zeros(shape=(len(t), n_vars))
    S = np.zeros(shape=(len(t), n_vars))
    S_ = np.zeros(shape=(len(t), n_vars))

    for i in range(n_vars):        
        beta_i = beta[i]
        gamma_i = gamma[i]
        t_i = t_[i]
        a_i = a[i]
        h_i = h[i]
        U[:, i], S[:, i], S_[:, i] = simulate_dynamics(
            t, a_i, h_i, t_i, beta_i, gamma_i, noise_model, noise_level
            )

    obs = {"true_t": t.round(2)}
    var = {
        "true_t_": t_[:n_vars],
        "true_beta": np.ones(n_vars) * beta,
        "true_gamma": np.ones(n_vars) * gamma,
        "true_scaling": np.ones(n_vars),
    }
    layers = {"unspliced": U, "spliced": S, "true_velocity": S_}

    return AnnData(S, obs, var, layers=layers)