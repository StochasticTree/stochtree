import numpy as np
import pandas as pd
import time
import sys
import os
from typing import Dict
from stochtree import BARTModel
from sklearn.model_selection import train_test_split

def dgp1(n: int, p: int, snr: float) -> Dict:
    rng = np.random.default_rng()
    
    # Covariates
    X = rng.uniform(0, 1, size=(n, p))
    
    # Piecewise linear term
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
    
    # Trigonometric term
    trig_term = 2 * np.sin(X[:, 2] * 2 * np.pi) - 1.5 * np.cos(X[:, 3] * 2 * np.pi)
    
    # Outcome
    f_XW = plm_term + trig_term
    noise_sd = np.std(f_XW) / snr
    y = f_XW + rng.normal(0, noise_sd, n)
    
    return {
        'covariates': X,
        'basis': None,
        'outcome': y,
        'conditional_mean': f_XW,
        'rfx_group_ids': None,
        'rfx_basis': None
    }


def dgp2(n: int, p: int, snr: float) -> Dict:
    rng = np.random.default_rng()
    
    # Covariates and basis
    X = rng.uniform(0, 1, (n, p))
    W = rng.uniform(0, 1, (n, 2))
    
    # Piecewise linear term using basis W
    plm_term = np.where(
        (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5 * W[:,0], 
        np.where(
            (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5 * W[:,0], 
            np.where(
                (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5 * W[:,0], 
                7.5 * W[:,0]
            )
        )
    )
    
    # Trigonometric term
    trig_term = 2 * np.sin(X[:, 2] * 2 * np.pi) - 1.5 * np.cos(X[:, 3] * 2 * np.pi)
    
    # Outcome
    f_XW = plm_term + trig_term
    noise_sd = np.std(f_XW) / snr
    y = f_XW + rng.normal(0, noise_sd, n)
    
    return {
        'covariates': X,
        'basis': W,
        'outcome': y,
        'conditional_mean': f_XW,
        'rfx_group_ids': None,
        'rfx_basis': None
    }


def dgp3(n: int, p: int, snr: float) -> Dict:
    rng = np.random.default_rng()
    
    # Covariates
    X = rng.uniform(0, 1, size=(n, p))
    
    # Piecewise linear term
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
    
    # Trigonometric term
    trig_term = 2 * np.sin(X[:, 2] * 2 * np.pi) - 1.5 * np.cos(X[:, 3] * 2 * np.pi)
    
    # Random effects
    num_groups = 3
    rfx_group_ids = rng.choice(num_groups, size=n)
    rfx_coefs = np.array([[-5, -3, -1], [5, 3, 1]]).T
    rfx_basis = np.column_stack([np.ones(n), np.random.uniform(-1, 1, n)])
    rfx_term = np.sum(rfx_coefs[rfx_group_ids] * rfx_basis, axis=1)
    
    # Outcome
    f_XW = plm_term + trig_term + rfx_term
    noise_sd = np.std(f_XW) / snr
    y = f_XW + rng.normal(0, noise_sd, n)
    
    return {
        'covariates': X,
        'basis': None,
        'outcome': y,
        'conditional_mean': f_XW,
        'rfx_group_ids': rfx_group_ids,
        'rfx_basis': rfx_basis
    }


def dgp4(n: int, p: int, snr: float) -> Dict:
    rng = np.random.default_rng()
    
    # Covariates and basis
    X = rng.uniform(0, 1, (n, p))
    W = rng.uniform(0, 1, (n, 2))
    
    # Piecewise linear term using basis W
    plm_term = np.where(
        (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5 * W[:,0], 
        np.where(
            (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5 * W[:,0], 
            np.where(
                (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5 * W[:,0], 
                7.5 * W[:,0]
            )
        )
    )
    
    # Trigonometric term
    trig_term = 2 * np.sin(X[:, 2] * 2 * np.pi) - 1.5 * np.cos(X[:, 3] * 2 * np.pi)
    
    # Random effects
    num_groups = 3
    rfx_group_ids = rng.choice(num_groups, size=n)
    rfx_coefs = np.array([[-5, -3, -1], [5, 3, 1]]).T
    rfx_basis = np.column_stack([np.ones(n), np.random.uniform(-1, 1, n)])
    rfx_term = np.sum(rfx_coefs[rfx_group_ids] * rfx_basis, axis=1)
    
    # Outcome
    f_XW = plm_term + trig_term + rfx_term
    noise_sd = np.std(f_XW) / snr
    y = f_XW + np.random.normal(0, noise_sd, n)
    
    return {
        'covariates': X,
        'basis': W,
        'outcome': y,
        'conditional_mean': f_XW,
        'rfx_group_ids': rfx_group_ids,
        'rfx_basis': rfx_basis
    }


def compute_test_train_indices(n: int, test_set_pct: float) -> Dict[str, np.ndarray]:
    sample_inds = np.arange(n)
    train_inds, test_inds = train_test_split(sample_inds, test_size=test_set_pct)
    return {'test_inds': test_inds, 'train_inds': train_inds}


def subset_data(data: np.ndarray, subset_inds: np.ndarray) -> np.ndarray:
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return data[subset_inds]
        else:
            return data[subset_inds, :]
    else:
        raise ValueError("Data must be a numpy array")


def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        n_iter = int(sys.argv[1])
        n = int(sys.argv[2])
        p = int(sys.argv[3])
        num_gfr = int(sys.argv[4])
        num_mcmc = int(sys.argv[5])
        dgp_num = int(sys.argv[6])
        snr = float(sys.argv[7])
        test_set_pct = float(sys.argv[8])
        num_threads = int(sys.argv[9])
    else:
        # Default arguments
        n_iter = 5
        n = 1000
        p = 5
        num_gfr = 10
        num_mcmc = 100
        dgp_num = 1
        snr = 2.0
        test_set_pct = 0.2
        num_threads = -1
    
    print(f"n_iter = {n_iter}")
    print(f"n = {n}")
    print(f"p = {p}")
    print(f"num_gfr = {num_gfr}")
    print(f"num_mcmc = {num_mcmc}")
    print(f"dgp_num = {dgp_num}")
    print(f"snr = {snr}")
    print(f"test_set_pct = {test_set_pct}")
    print(f"num_threads = {num_threads}")
    
    # Run the performance evaluation
    results = np.empty((n_iter, 4), dtype=float)
    
    for i in range(n_iter):
        print(f"Running iteration {i+1}/{n_iter}")
        
        # Generate data
        if dgp_num == 1:
            data_dict = dgp1(n=n, p=p, snr=snr)
        elif dgp_num == 2:
            data_dict = dgp2(n=n, p=p, snr=snr)
        elif dgp_num == 3:
            data_dict = dgp3(n=n, p=p, snr=snr)
        elif dgp_num == 4:
            data_dict = dgp4(n=n, p=p, snr=snr)
        else:
            raise ValueError("Invalid DGP input")
        
        covariates = data_dict['covariates']
        basis = data_dict['basis']
        conditional_mean = data_dict['conditional_mean']
        outcome = data_dict['outcome']
        rfx_group_ids = data_dict['rfx_group_ids']
        rfx_basis = data_dict['rfx_basis']
        
        # Split into train / test sets
        subset_inds_dict = compute_test_train_indices(n, test_set_pct)
        test_inds = subset_inds_dict['test_inds']
        train_inds = subset_inds_dict['train_inds']
        covariates_train = subset_data(covariates, train_inds)
        covariates_test = subset_data(covariates, test_inds)
        outcome_train = subset_data(outcome, train_inds)
        outcome_test = subset_data(outcome, test_inds)
        conditional_mean_train = subset_data(conditional_mean, train_inds)
        conditional_mean_test = subset_data(conditional_mean, test_inds)
        has_basis = basis is not None
        has_rfx = rfx_group_ids is not None
        if has_basis:
            basis_train = subset_data(basis, train_inds)
            basis_test = subset_data(basis, test_inds)
        else:
            basis_train = None
            basis_test = None
        if has_rfx:
            rfx_group_ids_train = subset_data(rfx_group_ids, train_inds)
            rfx_group_ids_test = subset_data(rfx_group_ids, test_inds)
            rfx_basis_train = subset_data(rfx_basis, train_inds)
            rfx_basis_test = subset_data(rfx_basis, test_inds)
        else:
            rfx_group_ids_train = None
            rfx_group_ids_test = None
            rfx_basis_train = None
            rfx_basis_test = None
        
        # Run (and time) BART
        start_time = time.time()
        
        # Sample BART model
        general_params = {'num_threads': num_threads}
        bart_model = BARTModel()
        bart_model.sample(
            X_train=covariates_train,
            y_train=outcome_train,
            leaf_basis_train=basis_train,
            rfx_group_ids_train=rfx_group_ids_train,
            rfx_basis_train=rfx_basis_train,
            num_gfr=num_gfr,
            num_mcmc=num_mcmc,
            general_params=general_params
        )
        
        # Predict on the test set
        test_preds = bart_model.predict(
            X=covariates_test,
            leaf_basis=basis_test,
            rfx_group_ids=rfx_group_ids_test,
            rfx_basis=rfx_basis_test
        )
        
        bart_timing = time.time() - start_time
        
        # Compute test set evaluations
        y_hat_posterior = test_preds['y_hat']
        y_hat_posterior_mean = np.mean(y_hat_posterior, axis=1)
        rmse_test = np.sqrt(np.mean((y_hat_posterior_mean - conditional_mean_test) ** 2))
        
        y_hat_posterior_quantile_025 = np.percentile(y_hat_posterior, 2.5, axis=1)
        y_hat_posterior_quantile_975 = np.percentile(y_hat_posterior, 97.5, axis=1)
        
        covered = np.logical_and(
            conditional_mean_test >= y_hat_posterior_quantile_025,
            conditional_mean_test <= y_hat_posterior_quantile_975
        )
        coverage_test = np.mean(covered)
        
        # Store evaluations
        results[i, :] = [i+1, rmse_test, coverage_test, bart_timing]
    
    # Wrangle and save results to CSV
    results_df = pd.DataFrame({
        'n': n,
        'p': p,
        'num_gfr': num_gfr,
        'num_mcmc': num_mcmc,
        'dgp_num': dgp_num,
        'snr': snr,
        'test_set_pct': test_set_pct,
        'num_threads': num_threads,
        'iter': results[:, 0],
        'rmse': results[:, 1],
        'coverage': results[:, 2],
        'runtime': results[:, 3]
    })
    
    snr_rounded = int(snr)
    test_set_pct_rounded = int(test_set_pct * 100)
    num_threads_clean = 0 if num_threads < 0 else num_threads
    filename = f"stochtree_bart_python_n_{n}_p_{p}_num_gfr_{num_gfr}_num_mcmc_{num_mcmc}_dgp_num_{dgp_num}_snr_{snr_rounded}_test_set_pct_{test_set_pct_rounded}_num_threads_{num_threads_clean}.csv"
    output_dir = "tools/regression/bart/stochtree_bart_python_results"
    filename_full = os.path.join(output_dir, filename)
    results_df.to_csv(filename_full, index=False)


if __name__ == "__main__":
    main() 