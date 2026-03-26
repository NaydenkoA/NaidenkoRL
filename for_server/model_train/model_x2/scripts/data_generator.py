import numpy as np

def simulate_gbm(n, dt, S0, mu, sigma, seed=None):
    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal(n)
    r1 = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z1
    S1 = np.empty(n + 1)
    S1[0] = S0
    S1[1:] = S0 * np.exp(np.cumsum(r1))

    return S1, r1

def simulate_4series(
    n, dt,
    S0,          # length-4 array of initial prices
    mu,          # length-4 annualized drift
    sigma,       # length-4 annualized vol
    rho24 = 0.65,       # target Corr(S2,S4) == Corr(S3,S4)
    k = 0.95,       # factor share for S2/S3 (0<k<=1)
    c = 0.7,       # S4 common-factor loading (0< c <1)
    seed=None
):

    rng = np.random.default_rng(seed)

    S0 = np.asarray(S0, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)

    r = np.sqrt(1/float(rho24)**2 - 1)
    a = k / np.sqrt(1 + r*r)
    b = r * a

    G = np.array([
        [0.0, 0.0],
        [a,  -b ],
        [a,   b ],
        [c,  0.0]
    ])

    norms2 = np.sum(G**2, axis=1)
    if np.any(norms2 >= 1):
        raise ValueError("Choose smaller k or c: some loading norm >=1")
    w_id = np.sqrt(1.0 - norms2)   # shape (4,)

    Z_common = rng.standard_normal((n, 2))  # factor shocks
    Z_idio   = rng.standard_normal((n, 4))  # idio shocks

    u = (Z_common @ G.T) + Z_idio * w_id

    drift = (mu - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)
    R = drift + u * vol      

    S = np.empty((n+1, 4))
    S[0] = S0
    S[1:] = S0 * np.exp(np.cumsum(R, axis=0))

    return S.T, R.T

def simulate_single_episode(num, sigma = 0.08522254220295641):
    S, _ = simulate_4series(
        num,
        1/250,
        [85000,20,20*85000,0.15],
        [0,0,0,0],
        [sigma, 0.1764965560805584, 0.17615395829118863, 0.10679331573472989]
    )

    s_calm, _ = simulate_gbm(
        n = num,
        dt = 1/250,
        S0 = 1,
        mu = 0,
        sigma = 0.001
    )

    vol_token_price = S[0]
    vol_token_balance = S[1]
    stable_token_balance = S[2]
    ts = S[3]

    volatile_part = vol_token_price*vol_token_balance/(vol_token_price*vol_token_balance + stable_token_balance)
    nav = (vol_token_price*vol_token_balance + stable_token_balance)/ts
    nect_price = s_calm

    return vol_token_price, nav, nect_price, volatile_part