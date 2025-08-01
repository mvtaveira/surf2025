import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# Damped Random Walk (Matthew's code)
def getDRWMag(tau, SFinf, mag_prev, dt):
  loc = mag_prev * np.exp(-dt / tau)
  scale = SFinf * np.sqrt((1. - np.exp(-2. * dt / tau)) / 2.)
  return loc + np.random.normal(0., scale)

def generateDRW(t, tau, SFinf = 0.3, xmean = 0, burn = 10000):
  # Generate in rest frame
  n = len(t)
  dt = np.diff(t)
  mag = np.zeros(n)
  mag[0] = np.random.normal(0, SFinf / np.sqrt(2.))
  for i in range(burn):
    mag[0] = getDRWMag(tau, SFinf, mag[0], dt[np.random.randint(n - 1)])
  for i in range(n - 1):
    mag[i + 1] = getDRWMag(tau, SFinf, mag[i], dt[i])
  return xmean + mag


# General time series generation
def generate_time_series(process: str, 
                         N: int = 1000,
                         amplitude: float = 1.0,        
                         mu_dt: float = 5.0,
                         sigma_dt: float = 0.3,  
                         drop_frac: float = 0.0,  
                         noise_sigma: float = 0.1, 
                         mag_err_sigma: float = 0.05,  
                         phi: float = 0.9,
                         n_cycles: int = 1,
                         # specific for DRW:
                         tau_drw: float = 50.0,
                         SFinf: float = 0.3,
                         burn: int = 10000,
                         seed: int = None) -> pd.DataFrame:
    
    """
    Generate synthetic time series for different signal types

    Parameters:
    - process: Type of base signal ('sin', 'square', 'random', 'red', 'DRW')
    - N: Number of data points
    - mu_dt: Mean time step
    - sigma_dt: Std. deviation of time step
    - drop_frac: Fraction of points to randomly drop to simulate gaps
    - noise_sigma: Noise added on top of clean signal
    - mag_err_sigma: Std. deviation of reported magnitude errors
    - phi: AR(1) coefficient for red noise
    - seed: Random seed for reproducibility

    Returns:
    - DataFRame with time, magnitude, and magnitude error
    """

    rng = np.random.default_rng(seed)
    dt = rng.normal(loc=mu_dt, scale=sigma_dt, size=N)
    time = np.cumsum(dt)

    # Randomly drop some timestamps to simulate observational gaps
    if drop_frac > 0:
        mask = rng.random(time.shape[0]) > drop_frac
        time = time[mask]   


    t_norm = 2 * np.pi * n_cycles* (time - time.min()) / (time.max() - time.min()) 

    # Generate base signal
    if process == 'sin': 
        mag_base = np.sin(t_norm)
    elif process == 'square':
        mag_base = np.sign(np.sin(t_norm))
    elif process == 'random':   # white noise
        mag_base = rng.normal(0, 1, size=len(time))
    elif process == 'red':
        mag_base = np.zeros(len(time))
        mag_base[0] = rng.normal(0,1)
        for i in range (1, len(time)):
            mag_base[i] = phi * mag_base[i-1] + rng.normal(0,1)
    elif process == 'drw':
        mag_base = generateDRW(time, tau=tau_drw, SFinf=SFinf, burn=burn)

    else:
        raise ValueError("process must be one of: 'sin', 'square', 'random', 'red', 'drw'")
    
    if process != 'drw':
    # Add Gaussian observational noise to the signal
        mag = amplitude * mag_base + rng.normal(0, noise_sigma, size=len(time))
    else:
        mag = amplitude * mag_base
    
    # Generate magnitude errors
    mag_err = np.abs(rng.normal(0, mag_err_sigma, size=len(time)))

    return pd.DataFrame({'time': time, 'mag': mag, 'mag_err': mag_err})


# Time Delay Embedding (Takens Embedding)
def embed_series(x, m, tau): 
    """
    Perform Takens' time-delay embedding of a 1D time series

    Parameters:
    - x: 1D time series array
    - m: Embedding dimension
    - tau: Time delay between embedded components

    Returns:
    - 2D array of shape (M, m), the embedded vectors
    """

    N = len(x)
    M = N - (m-1) * tau  
    return np.array([x[i : i + m * tau : tau] for i in range(M)])


# Recurrence Plot
def recurrence_plot(df, m=2, tau=1, eps=None, perc=None, save_path=None, show=False):
    """
    Compute and display a recurrence plot of a time series

    Parameters:
    - df: DataFrame with 'mag' column representing the time series
    - m: Embedding dimension
    - tau: Time delay
    - eps: Distance threshold for recurrence (optional)
    - perc: Percentile to determine eps if not explicitly set

    Returns:
    - Binary recurrence matrix (1 for recurrence, 0 otherwise)
    """

    x = df['mag'].values
    emb = embed_series(x, m, tau)

    # Compute pairwise Euclidean distances between embedded vectors
    D = np.sqrt(((emb[:, None, :] - emb[None, :, :])**2).sum(axis=2))

    # Set threshold distance for recurrence based on percentile or median
    if eps is None:
        if perc is not None:
            eps = np.percentile(D, perc)
        else:
            eps = np.median(D)

    # Binary recurrence matrix
    R = (D <= eps).astype(int)

    plt.figure(figsize=(6,6))
    plt.imshow(R, origin='lower', cmap='Greys', interpolation='none')
    plt.xlabel('Embedding index')
    plt.ylabel('Embedding index')
    plt.title(r'Recurrence Plot ($m$=%d, $\tau$=%d, $\epsilon$=%.2f)' % (m, tau, eps))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    elif show:
        plt.show()
    else:
        plt.close()

    return R, eps
