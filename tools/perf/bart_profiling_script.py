# Load libraries
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from stochtree import BARTModel
import time

def outcome_mean(X: np.ndarray) -> np.ndarray:
    plm_term = np.where(
        (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5 * X[:,1], 
        np.where(
            (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5 * X[:,1], 
            np.where(
                (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5 * X[:,1], 
                7.5 * X[:,1]
            )
        )
    )
    trig_term = 2*np.sin(X[:,2]*2*np.pi) - 1.5*np.cos(X[:,3]*2*np.pi)
    return plm_term + trig_term

if __name__ == "__main__":
    # Handle optional command line arguments
    parser = argparse.ArgumentParser(
                    prog='bart_profiling_script',
                    description='Runs BART on synthetic data, following user-provided parameters')
    parser.add_argument("--n", action='store', default=1000, type=int)
    parser.add_argument("--p", action='store', default=5, type=int)
    parser.add_argument("--num_gfr", action='store', default=10, type=int)
    parser.add_argument("--num_mcmc", action='store', default=100, type=int)
    parser.add_argument("--snr", action='store', default=2.0, type=float)
    parser.add_argument("--num_threads", action='store', default=-1, type=int)
    args = parser.parse_args()
    n = args.n
    p = args.p
    num_gfr = args.num_gfr
    num_mcmc = args.num_mcmc
    snr = args.snr
    num_threads = args.num_threads
    print(f"n = {n:d}\np = {p:d}\nnum_gfr = {num_gfr:d}\nnum_mcmc = {num_mcmc:d}\nsnr = {snr:.2f}\nnum_threads = {num_threads:d}")

    # Generate synthetic data
    rng = np.random.default_rng()
    X = rng.uniform(0, 1, (n, p))
    f_X = outcome_mean(X)
    noise_sd = np.std(f_X)/snr
    epsilon = rng.normal(loc=0., scale=noise_sd, size=n)
    y = f_X + epsilon

    # Test train split
    sample_inds = np.arange(n)
    train_inds, test_inds = train_test_split(sample_inds, test_size=0.2)
    X_train = X[train_inds,:]
    X_test = X[test_inds,:]
    y_train = y[train_inds]
    y_test = y[test_inds]

    # Time the BART model and prediction
    start_time = time.time()
    general_params = {'num_threads': num_threads}
    bart_model = BARTModel()
    bart_model.sample(X_train=X_train, y_train=y_train, X_test=X_test, 
                      num_gfr=num_gfr, num_mcmc=num_mcmc, general_params=general_params)
    bart_preds = bart_model.predict(covariates=X_test)
    test_preds = bart_preds['y_hat']
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"Total runtime: {total_runtime:.3f} seconds")
