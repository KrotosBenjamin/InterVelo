import warnings
import numpy as np
from anndata import AnnData
from scvelo.core import invert
import random


def chromatin(tau, c0, k_c, alpha_c):
    """TODO."""
    expc = np.exp(-alpha_c * tau)
    return c0 * expc + k_c * (1 - expc)


def unspliced(tau, u0, c0, k_c,alpha_c, alpha, beta):
    """TODO."""
    expu = np.exp(-beta * tau)
    expc = np.exp(-alpha_c * tau)
    return u0 * expu + alpha * k_c / beta * (1 - expu) + (k_c-c0) * alpha / (beta - alpha)*(expu - expc)


# TODO: Add docstrings
def spliced(tau, s0, u0, c0, k_c, alpha_c, alpha, beta, gamma):
    """TODO."""
    c = (alpha * k_c / beta - u0 - (k_c-c0) * alpha / (beta-alpha)) * invert(gamma - beta) * beta
    d = invert(gamma - alpha) * beta * (k_c-c0) * alpha * invert(beta - alpha)
    expu, exps = np.exp(-beta * tau), np.exp(-gamma * tau)
    expc = np.exp(-alpha_c * tau)
    return s0 * exps + alpha * k_c / gamma * (1 - exps) + c * (exps - expu) + d * (exps - expc)


# TODO: Add docstrings
def vectorize(model, t, t_, t_max, alpha, alpha_c, beta, gamma=None, alpha_=0, u0=0, s0=0,c0=0, sorted=False):
    """TODO."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o1 = np.array(t < t_ * 0.5, dtype=int)
        o2 = np.array((t_ * 0.5 <= t) & (t < t_), dtype=int)
        o3 = np.array((t_ <= t) & (t < 0.5 * t_ + 0.5 * t_max), dtype=int)
        o4 = 1 - o1 -o2 -o3

    tau = t * o1 + (t - t_*0.5) * o2 + (t - t_) * o3 + (t- t_ * 0.5 - t_max * 0.5) * o4
    
    if model==1:
        u0_1 = u0
        s0_1 = s0
        c0_1 = chromatin(t_*0.5, c0, 1, alpha_c)
        u0_2 = unspliced(t_*0.5, u0_1, c0_1, 1, alpha_c, alpha, beta)
        s0_2 = spliced(t_*0.5, s0_1, u0_1, c0_1, 1, alpha_c, alpha, beta, gamma if gamma is not None else beta / 2)
        c0_2 = chromatin(t_*0.5, c0_1, 1, alpha_c)
        u0_3 = unspliced((t_max-t_)*0.5, u0_2, c0_2, 0, alpha_c, alpha, beta)
        s0_3 = spliced((t_max-t_)*0.5, s0_2, u0_2, c0_2, 0, alpha_c, alpha, beta, gamma if gamma is not None else beta / 2)
        c0_3 = chromatin((t_max-t_)*0.5, c0_2, 0, alpha_c)
    else:
        u0_1 = u0
        s0_1 = s0
        c0_1 = chromatin(t_*0.5, c0, 1, alpha_c)
        u0_2 = unspliced(t_*0.5, u0_1, c0_1, 1, alpha_c, alpha, beta)
        s0_2 = spliced(t_*0.5, s0_1, u0_1, c0_1, 1, alpha_c, alpha, beta, gamma if gamma is not None else beta / 2)
        c0_2 = chromatin(t_*0.5, c0_1, 1, alpha_c)
        u0_3 = unspliced((t_max-t_)*0.5, u0_2, c0_2, 1, alpha_c, alpha_, beta)
        s0_3 = spliced((t_max-t_)*0.5, s0_2, u0_2, c0_2, 1, alpha_c, alpha_, beta, gamma if gamma is not None else beta / 2)
        c0_3 = chromatin((t_max-t_)*0.5, c0_2, 1, alpha_c)

    # vectorize u0, s0 and alpha
    u0 = u0 * o1 + u0_1 * o2 + u0_2 * o3 + u0_3 * o4
    s0 = s0 * o1 + s0_1 * o2 + s0_2 * o3 + s0_3 * o4
    c0 = c0 * o1 + c0_1 * o2 + c0_2 * o3 + c0_3 * o4
    if model==1:
        alpha = alpha_ * o1 + alpha * o2 + alpha * o3 + alpha_ * o4
        k_c = o1 + o2
    else:
        alpha = alpha_ * o1 + alpha * o2 + alpha_ * o3 + alpha_ * o4
        k_c = o1 + o2 + o3

    if sorted:
        idx = np.argsort(t)
        tau, alpha, u0, s0 = tau[idx], alpha[idx], u0[idx], s0[idx]
    
    u0 = np.clip(u0,0,None)
    s0 = np.clip(s0,0,None)
    c0 = np.clip(c0,0,None)

    return tau, alpha, u0, s0, c0, k_c


def simulation(
    n_obs=300,
    n_vars=None,
    alpha_c=None,
    alpha=None,
    beta=None,
    gamma=None,
    alpha_=None,
    t_max=None,
    noise_model="normal",
    noise_level=1,
    random_seed=0,
):
    """Simulation of mRNA splicing kinetics.

    Simulated DNA accessibility and mRNA metabolism.
    The parameters for each reaction are randomly sampled from a log-normal distribution
    and time events follow the Poisson law. The total time spent in a transcriptional
    state is varied between two and ten hours.

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

    def simulate_dynamics(tau, alpha, alpha_c, beta, gamma, u0, s0, c0, k_c, noise_model, noise_level):
        ut = unspliced(tau, u0, c0, k_c,alpha_c, alpha, beta)
        st = spliced(tau, s0, u0, c0, k_c, alpha_c, alpha, beta, gamma)
        ct = chromatin(tau, c0, k_c, alpha_c)
        if noise_model == "normal":  # add noise
            ut += np.random.normal(
                scale=noise_level * np.percentile(ut, 99) / 10, size=len(ut)
            )
            st += np.random.normal(
                scale=noise_level * np.percentile(st, 99) / 10, size=len(st)
            )
            ct += np.random.normal(
                scale=noise_level * np.percentile(ct, 99) / 10, size=len(ct)
            )
        ut, st, ct = np.clip(ut, 0, None), np.clip(st, 0, None),np.clip(ct, 0, None)
        return ut, st, ct

    alpha = 5 if alpha is None else alpha
    alpha_c = 0.5  if alpha_c is None else alpha_c
    beta = 0.5 if beta is None else beta
    gamma = 0.3 if gamma is None else gamma
    alpha_ = 0 if alpha_ is None else alpha_

    t = draw_poisson(n_obs)
    if t_max is not None:
        t *= t_max / np.max(t)
    t_max = np.max(t)

    def cycle(array, n_vars=None):
        if isinstance(array, (np.ndarray, list, tuple)):
            return (
                array if n_vars is None else array * int(np.ceil(n_vars / len(array)))
            )
        else:
            return [array] if n_vars is None else [array] * n_vars
    
    # switching time point obtained as fraction of t_max rounded down
    switches = np.random.uniform(0.1,0.5,n_vars)
    t_ = np.array([np.max(t[t < t_i * t_max]) for t_i in switches])

    model1 = random.sample(list(range(n_vars)), int(np.floor(n_vars*0.5)))

    noise_level = cycle(noise_level, len(switches) if n_vars is None else n_vars)

    n_vars = min(len(switches), len(noise_level)) if n_vars is None else n_vars
    U = np.zeros(shape=(len(t), n_vars))
    S = np.zeros(shape=(len(t), n_vars))
    C = np.zeros(shape=(len(t), n_vars))

    def is_list(x):
        return isinstance(x, (tuple, list, np.ndarray))

    for i in range(n_vars):
        alpha_i = alpha[i] if is_list(alpha) and len(alpha) != n_obs else alpha
        alpha_c_i = alpha_c[i] if is_list(alpha_c) and len(alpha_c) != n_obs else alpha_c
        beta_i = beta[i] if is_list(beta) and len(beta) != n_obs else beta
        gamma_i = gamma[i] if is_list(gamma) and len(gamma) != n_obs else gamma
        model=1 if i in model1 else 2
        tau, alpha_vec, u0_vec, s0_vec, c0_vec, k_c_vec = vectorize(
            model, t, t_[i], t_max, alpha_i, alpha_c_i, beta_i, gamma_i, alpha_=alpha_, u0=0, s0=0, c0=0
        )
        
        U[:, i], S[:, i], C[:, i] = simulate_dynamics(
                tau,
                alpha_vec,
                alpha_c_i,
                beta_i,
                gamma_i,
                u0_vec,
                s0_vec,
                c0_vec,
                k_c_vec,
                noise_model,
                noise_level[i],
            )

    
    if is_list(alpha) and len(alpha) == n_obs:
        alpha = np.nan
    if is_list(beta) and len(beta) == n_obs:
        beta = np.nan
    if is_list(gamma) and len(gamma) == n_obs:
        gamma = np.nan

    obs = {"true_t": t.round(2)}
    var = {
        "true_t_": t_[:n_vars],
        "true_alpha": np.ones(n_vars) * alpha,
        "true_beta": np.ones(n_vars) * beta,
        "true_gamma": np.ones(n_vars) * gamma,
        "true_scaling": np.ones(n_vars),
    }
    layers = {"unspliced": U, "spliced": S, "atac": C}

    return AnnData(S, obs, var, layers=layers)