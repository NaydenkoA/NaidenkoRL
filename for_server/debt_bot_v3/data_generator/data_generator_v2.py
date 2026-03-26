import numpy as np
import pandas as pd
from statsmodels.iolib.smpickle import load_pickle

GARCH = {
    "S1": {"omega": 0.02, "alpha": [0.05], "beta": [0.93], "nu": 4.06, "scale": 0.96761, "dist": "t"},
    "S2": {"omega": 0.1, "alpha": [0.1], "beta": [0.8], "nu": 4.01, "scale": 0.97875, "dist": "t"},
    "S3": {"omega": 0.1, "alpha": [0.05], "beta": [0.85], "nu": 4.01, "scale": 0.98, "dist": "t"},
    "S4": {"omega": 0.01, "alpha": [0.01], "beta": [0.98], "scale": 1, "dist": "normal"}
}

SIGMA = {
    "S1": 0.0004868141613705003,
    "S2": 0.002859470286952047,
    "S3": 0.002641502701385392,
    "S4": 0.001851549810493781 
}

def simulate_garch_eps(garch_params, cov_resid, n_steps, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    names = list(garch_params.keys())
    k = len(names)

    cov_resid = np.asarray(cov_resid)
    std_cov = np.sqrt(np.diag(cov_resid))
    corr_resid = cov_resid / np.outer(std_cov, std_cov)
    np.fill_diagonal(corr_resid, 1.0)

    L = np.linalg.cholesky(corr_resid)

    omega = np.array([float(garch_params[n]['omega']) for n in names])
    alpha_list = [np.array(garch_params[n].get('alpha', []), dtype=float) for n in names]
    beta_list  = [np.array(garch_params[n].get('beta', []),  dtype=float) for n in names]
    dist  = [garch_params[n].get('dist', 'normal') for n in names]
    nu    = [garch_params[n].get('nu', None) for n in names]
    scale = np.array([float(garch_params[n].get('scale', 1.0)) for n in names])

    max_p = max(len(a) for a in alpha_list) or 1
    max_q = max(len(b) for b in beta_list) or 1
    burn_in = max(max_p, max_q, 10)
    total_steps = n_steps + burn_in

    eps = np.zeros((total_steps, k))
    sigma = np.ones((total_steps, k))

    for t in range(1, total_steps):
        for i in range(k):
            alpha = alpha_list[i]
            beta  = beta_list[i]
            var_t = omega[i]

            for j, a in enumerate(alpha):
                lag = t - 1 - j
                if lag >= 0:
                    var_t += a * eps[lag, i] ** 2

            for j, b in enumerate(beta):
                lag = t - 1 - j
                if lag >= 0:
                    var_t += b * sigma[lag, i] ** 2

            sigma[t, i] = np.sqrt(max(var_t, 1e-8))

        z = np.zeros(k)
        for i in range(k):
            if dist[i] == "t" and nu[i] is not None:
                df = float(nu[i])
                z[i] = rng.standard_t(df) / np.sqrt(df / (df - 2))
            else:
                z[i] = rng.standard_normal()

        eps[t, :] = (L @ z) * sigma[t, :]

    eps = eps[burn_in:]
    sigma = sigma[burn_in:]

    for i in range(k):
        if scale[i] != 1.0:
            eps[:, i] *= scale[i]

    return eps, sigma


def simulate_var_garch(var_results, garch_params, n_steps, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    names = var_results.names
    k = len(names)
    p = var_results.k_ar

    eps, sigma = simulate_garch_eps(garch_params, var_results.sigma_u, n_steps + p, rng=rng)

    y = np.zeros((n_steps + p, k))

    A = [var_results.params.loc[f"L{lag}.S1":f"L{lag}.S4"].values.T for lag in range(1, p + 1)]
    const = var_results.params.loc["const"].values if "const" in var_results.params.index else np.zeros(k)

    for t in range(p, n_steps + p):
        y_t = const.copy()
        for lag in range(1, p + 1):
            y_t += A[lag - 1] @ y[t - lag, :]
        y[t, :] = y_t + eps[t, :]

    y_df = pd.DataFrame(y[p:], columns=names)
    return y_df

def simulate_gbm(n, dt, S0, mu, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    Z1 = rng.standard_normal(n)
    r1 = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z1
    S1 = np.empty(n + 1)
    S1[0] = S0
    S1[1:] = S0 * np.exp(np.cumsum(r1))

    return S1, r1

def simulate_data(n_steps, S0 = [85000, 30, 3.5e6, 0.1], rng = None, sigma_token = None, sigma_amounts = None):
    var_results = load_pickle("wbtc_island_var_model.pkl")

    df = simulate_var_garch(
        var_results,
        GARCH,
        n_steps, 
        rng
    )

    s1 = df['S1'].values
    s2 = df['S2'].values
    s3 = df['S3'].values
    s4 = df['S4'].values

    s1_normed = s1/np.std(s1)
    s2_normed = s2/np.std(s2)
    s3_normed = s3/np.std(s3)
    s4_normed = s4/np.std(s4)

    if sigma_token is None:
        sigma1 = SIGMA["S1"]
    else:
        sigma1 = sigma_token

    if sigma_amounts is None:
        sigma2 = SIGMA["S2"]
        sigma3 = SIGMA["S3"]
    else:
        sigma2 = sigma_amounts
        sigma3 = sigma_amounts
    
    sigma4 = SIGMA["S4"]

    p = S0[0]*np.exp(sigma1*np.cumsum(s1_normed))
    v = S0[1]*np.exp(sigma2*np.cumsum(s2_normed))
    s = S0[2]*np.exp(sigma3*np.cumsum(s3_normed))
    ts = S0[3]*np.exp(sigma4*np.cumsum(s4_normed))

    nav = (p*v + s)/ts
    vol_part = p*v/(p*v + s)

    p_stable, _ = simulate_gbm(
        n_steps,
        1/252,
        1,
        0,
        0.00001,
        rng
    )

    return p, nav, vol_part, p_stable

