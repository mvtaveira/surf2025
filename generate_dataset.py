import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from rp_generator import generate_time_series, recurrence_plot

def generate_dataset(signal_type='sin',
                     num_samples=100,
                     output_dir='dataset',
                     tau_drw=50.0,
                     SFinf=0.3,
                     burn=10000,
                     m_fixed=None,
                     tau_fixed=None,
                     perc_fixed=None,
                     amplitude_range=(0.5, 2.0),
                     drop_frac_range=(0.0, 0.5),
                     noise_sigma_range=(0.0, 0.2),
                     cycles_range=(1,1)
                     ):  
    
    rng = np.random.default_rng()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/arrays", exist_ok=True)
    
    metadata = []
    for i in range(num_samples):
        seed = i

        amp       = float(rng.uniform(*amplitude_range))
        drop_frac = float(rng.uniform(*drop_frac_range))
        noise_sigma = rng.uniform(*noise_sigma_range)

        n_cycles = int(rng.integers(cycles_range[0], cycles_range[1] + 1))

        m = m = m_fixed if m_fixed is not None else np.random.randint(2, 11)
        perc = perc_fixed if perc_fixed is not None else np.random.choice([5, 10, 15, 20])
        
        if tau_fixed is not None:
            tau = int(tau_fixed)
        elif signal_type == 'drw':
            tau = int(tau_drw)
        else:
            tau = np.random.randint(2, 61)

        df = generate_time_series(signal_type, 
                                  seed=seed, 
                                  tau_drw=tau_drw, 
                                  SFinf=SFinf, 
                                  burn=burn, 
                                  amplitude=amp, 
                                  drop_frac=drop_frac, 
                                  noise_sigma=noise_sigma, 
                                  n_cycles=n_cycles)
        R, eps = recurrence_plot(df, m=m, tau=tau, perc=perc, show=False)

        # Save image
        img_path = f"{output_dir}/images/{signal_type}_{i:03d}.png"
        plt.imsave(img_path, R, origin='lower', cmap='Greys')

        # Save array
        npy_path = f"{output_dir}/arrays/{signal_type}_{i:03d}.npy"
        np.save(npy_path, R)

        # Save metadata
        metadata.append({
            'index': i,
            'signal_type': signal_type,
            'm': m,
            'tau': tau,
            'eps': eps,
            'amplitude': amp,
            'drop_frac': drop_frac,
            'noise_sigma': noise_sigma,
            'perc': perc,
            'n_cycles': n_cycles,
            'seed': seed,
            'img_path': img_path,
            'npy_path': npy_path
        })
    
    df_meta = pd.DataFrame(metadata)
    meta_csv_path = f"{output_dir}/metadata.csv"
    df_meta.to_csv(meta_csv_path, index=False)
    return df_meta