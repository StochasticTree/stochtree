import numpy as np
import pandas as pd
import time
import sys
import os
from typing import Dict
from stochtree import BCFModel
from sklearn.model_selection import train_test_split
from scipy.stats import norm


def dgp1(n: int, p: int, snr: float) -> Dict:
    rng = np.random.default_rng()
    
    # Covariates
    X = rng.normal(0, 1, size=(n, p))
    
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
    trig_term = 2 * np.sin(X[:, 2] * 2 * np.pi) - 2 * np.cos(X[:, 3] * 2 * np.pi)
    
    # Prognostic effect
    mu_x = plm_term + trig_term
    
    # Propensity score
    pi_x = 0.8 * norm.cdf((3 * mu_x / np.std(mu_x)) - 0.5 * X[:, 0]) + 0.05 + rng.uniform(0, 1, n) / 10
    
    # Treatment assignment
    Z = rng.binomial(1, pi_x, n)
    
    # Treatment effect
    tau_x = 1 + 2 * X[:, 1] * X[:, 3]
    
    # Outcome
    f_XZ = mu_x + tau_x * Z
    noise_sd = np.std(f_XZ) / snr
    y = f_XZ + rng.normal(0, noise_sd, n)
    
    return {
        'covariates': X,
        'treatment': Z,
        'outcome': y,
        'propensity': pi_x,
        'prognostic_effect': mu_x,
        'treatment_effect': tau_x,
        'conditional_mean': f_XZ,
        'rfx_group_ids': None,
        'rfx_basis': None
    }


def dgp2(n: int, p: int, snr: float) -> Dict:
    rng = np.random.default_rng()
    
    # Covariates
    X = rng.uniform(0, 1, size=(n, p))
    
    # Propensity scores (multivariate)
    pi_x = np.column_stack([
        0.125 + 0.75 * X[:, 0],
        0.875 - 0.75 * X[:, 1]
    ])
    
    # Prognostic effect
    mu_x = pi_x[:, 0] * 5 + pi_x[:, 1] * 2 + 2 * X[:, 2]
    
    # Treatment effects (multivariate)
    tau_x = np.column_stack([
        X[:, 1] * 2,
        X[:, 2] * 2
    ])
    
    # Treatment assignment (multivariate)
    Z = np.column_stack([
        rng.binomial(1, pi_x[:, 0], n),
        rng.binomial(1, pi_x[:, 1], n)
    ])
    
    # Outcome
    f_XZ = mu_x + np.sum(Z * tau_x, axis=1)
    noise_sd = np.std(f_XZ) / snr
    y = f_XZ + rng.normal(0, noise_sd, n)
    
    return {
        'covariates': X,
        'treatment': Z,
        'outcome': y,
        'propensity': pi_x,
        'prognostic_effect': mu_x,
        'treatment_effect': tau_x,
        'conditional_mean': f_XZ,
        'rfx_group_ids': None,
        'rfx_basis': None
    }


def dgp3(n: int, p: int, snr: float) -> Dict:
    rng = np.random.default_rng()
    
    # Covariates
    X = rng.normal(0, 1, size=(n, p))
    
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
    trig_term = 2 * np.sin(X[:, 2] * 2 * np.pi) - 2 * np.cos(X[:, 3] * 2 * np.pi)
    
    # Prognostic effect
    mu_x = plm_term + trig_term
    
    # Propensity score
    pi_x = 0.8 * norm.cdf((3 * mu_x / np.std(mu_x)) - 0.5 * X[:, 0]) + 0.05 + rng.uniform(0, 1, n) / 10
    
    # Treatment assignment
    Z = rng.binomial(1, pi_x, n)
    
    # Treatment effect
    tau_x = 1 + 2 * X[:, 1] * X[:, 3]
    
    # Random effects
    rfx_group_ids = rng.choice([0, 1, 2], size=n, replace=True)
    rfx_coefs = np.array([[-5, -3, -1], [5, 3, 1]]).T
    rfx_basis = np.column_stack([np.ones(n), rng.uniform(-1, 1, n)])
    rfx_term = np.sum(rfx_coefs[rfx_group_ids] * rfx_basis, axis=1)
    
    # Outcome
    f_XZ = mu_x + tau_x * Z + rfx_term
    noise_sd = np.std(f_XZ) / snr
    y = f_XZ + rng.normal(0, noise_sd, n)
    
    return {
        'covariates': X,
        'treatment': Z,
        'outcome': y,
        'propensity': pi_x,
        'prognostic_effect': mu_x,
        'treatment_effect': tau_x,
        'conditional_mean': f_XZ,
        'rfx_group_ids': rfx_group_ids,
        'rfx_basis': rfx_basis
    }


def dgp4(n: int, p: int, snr: float) -> Dict:
    rng = np.random.default_rng()
    
    # Covariates
    X = rng.uniform(0, 1, size=(n, p))
    
    # Propensity scores (multivariate)
    pi_x = np.column_stack([
        0.125 + 0.75 * X[:, 0],
        0.875 - 0.75 * X[:, 1]
    ])
    
    # Prognostic effect
    mu_x = pi_x[:, 0] * 5 + pi_x[:, 1] * 2 + 2 * X[:, 2]
    
    # Treatment effects (multivariate)
    tau_x = np.column_stack([
        X[:, 1] * 2,
        X[:, 2] * 2
    ])
    
    # Treatment assignment (multivariate)
    Z = np.column_stack([
        rng.binomial(1, pi_x[:, 0], n),
        rng.binomial(1, pi_x[:, 1], n)
    ])
    
    # Random effects
    rfx_group_ids = rng.choice([0, 1, 2], size=n, replace=True)
    rfx_coefs = np.array([[-5, -3, -1], [5, 3, 1]]).T
    rfx_basis = np.column_stack([np.ones(n), rng.uniform(-1, 1, n)])
    rfx_term = np.sum(rfx_coefs[rfx_group_ids] * rfx_basis, axis=1)
    
    # Outcome
    f_XZ = mu_x + np.sum(Z * tau_x, axis=1) + rfx_term
    noise_sd = np.std(f_XZ) / snr
    y = f_XZ + rng.normal(0, noise_sd, n)
    
    return {
        'covariates': X,
        'treatment': Z,
        'outcome': y,
        'propensity': pi_x,
        'prognostic_effect': mu_x,
        'treatment_effect': tau_x,
        'conditional_mean': f_XZ,
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
    results = np.empty((n_iter, 6), dtype=float)
    
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
        treatment = data_dict['treatment']
        propensity = data_dict['propensity']
        prognostic_effect = data_dict['prognostic_effect']
        treatment_effect = data_dict['treatment_effect']
        conditional_mean = data_dict['conditional_mean']
        outcome = data_dict['outcome']
        rfx_group_ids = data_dict['rfx_group_ids']
        rfx_basis = data_dict['rfx_basis']
        
        # Check if multivariate treatment
        has_multivariate_treatment = dgp_num in [2, 4]
        
        # Split into train / test sets
        subset_inds_dict = compute_test_train_indices(n, test_set_pct)
        test_inds = subset_inds_dict['test_inds']
        train_inds = subset_inds_dict['train_inds']
        covariates_train = subset_data(covariates, train_inds)
        covariates_test = subset_data(covariates, test_inds)
        treatment_train = subset_data(treatment, train_inds)
        treatment_test = subset_data(treatment, test_inds)
        propensity_train = subset_data(propensity, train_inds)
        propensity_test = subset_data(propensity, test_inds)
        outcome_train = subset_data(outcome, train_inds)
        outcome_test = subset_data(outcome, test_inds)
        prognostic_effect_train = subset_data(prognostic_effect, train_inds)
        prognostic_effect_test = subset_data(prognostic_effect, test_inds)
        treatment_effect_train = subset_data(treatment_effect, train_inds)
        treatment_effect_test = subset_data(treatment_effect, test_inds)
        conditional_mean_train = subset_data(conditional_mean, train_inds)
        conditional_mean_test = subset_data(conditional_mean, test_inds)
        has_rfx = rfx_group_ids is not None
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
        
        # Run (and time) BCF
        start_time = time.time()
        
        # Sample BCF model
        general_params = {'num_threads': num_threads, 'adaptive_coding': False}
        prognostic_forest_params = {'sample_sigma2_leaf': False}
        treatment_effect_forest_params = {'sample_sigma2_leaf': False}
        
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=covariates_train,
            Z_train=treatment_train,
            y_train=outcome_train,
            pi_train=propensity_train,
            rfx_group_ids_train=rfx_group_ids_train,
            rfx_basis_train=rfx_basis_train,
            num_gfr=num_gfr,
            num_mcmc=num_mcmc,
            general_params=general_params,
            prognostic_forest_params=prognostic_forest_params,
            treatment_effect_forest_params=treatment_effect_forest_params
        )
        
        # Predict on the test set
        test_preds = bcf_model.predict(
            X=covariates_test,
            Z=treatment_test,
            propensity=propensity_test,
            rfx_group_ids=rfx_group_ids_test,
            rfx_basis=rfx_basis_test
        )
        
        bcf_timing = time.time() - start_time
        
        # Compute test set evaluations
        y_hat_posterior = test_preds['y_hat']
        tau_hat_posterior = test_preds['tau_hat']
        
        y_hat_posterior_mean = np.mean(y_hat_posterior, axis=1)
        if has_multivariate_treatment:
            # For multivariate treatment, tau_hat_posterior has shape (n_test, n_samples, n_treatments)
            # We want to average over the samples (axis 1) to get (n_test, n_treatments)
            tau_hat_posterior_mean = np.mean(tau_hat_posterior, axis=1)
        else:
            # For univariate treatment, tau_hat_posterior has shape (n_test, n_samples)
            # We want to average over the samples (axis 1) to get (n_test,)
            tau_hat_posterior_mean = np.mean(tau_hat_posterior, axis=1)
        
        # Outcome RMSE and coverage
        y_hat_rmse_test = np.sqrt(np.mean((y_hat_posterior_mean - conditional_mean_test) ** 2))
        y_hat_posterior_quantile_025 = np.percentile(y_hat_posterior, 2.5, axis=1)
        y_hat_posterior_quantile_975 = np.percentile(y_hat_posterior, 97.5, axis=1)
        
        y_hat_covered = np.logical_and(
            conditional_mean_test >= y_hat_posterior_quantile_025,
            conditional_mean_test <= y_hat_posterior_quantile_975
        )
        y_hat_coverage_test = np.mean(y_hat_covered)
        
        # Treatment effect RMSE and coverage
        tau_hat_rmse_test = np.sqrt(np.mean((tau_hat_posterior_mean - treatment_effect_test) ** 2))
        
        if has_multivariate_treatment:
            # For multivariate treatment, compute percentiles over samples (axis 1)
            tau_hat_posterior_quantile_025 = np.percentile(tau_hat_posterior, 2.5, axis=1)
            tau_hat_posterior_quantile_975 = np.percentile(tau_hat_posterior, 97.5, axis=1)
            tau_hat_covered = np.logical_and(
                treatment_effect_test >= tau_hat_posterior_quantile_025,
                treatment_effect_test <= tau_hat_posterior_quantile_975
            )
        else:
            # For univariate treatment, compute percentiles over samples (axis 1)
            tau_hat_posterior_quantile_025 = np.percentile(tau_hat_posterior, 2.5, axis=1)
            tau_hat_posterior_quantile_975 = np.percentile(tau_hat_posterior, 97.5, axis=1)
            tau_hat_covered = np.logical_and(
                treatment_effect_test >= tau_hat_posterior_quantile_025,
                treatment_effect_test <= tau_hat_posterior_quantile_975
            )
        
        tau_hat_coverage_test = np.mean(tau_hat_covered)
        
        # Store evaluations
        results[i, :] = [i+1, y_hat_rmse_test, y_hat_coverage_test, tau_hat_rmse_test, tau_hat_coverage_test, bcf_timing]
    
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
        'outcome_rmse': results[:, 1],
        'outcome_coverage': results[:, 2],
        'treatment_effect_rmse': results[:, 3],
        'treatment_effect_coverage': results[:, 4],
        'runtime': results[:, 5]
    })
    
    snr_rounded = int(snr)
    test_set_pct_rounded = int(test_set_pct * 100)
    num_threads_clean = 0 if num_threads < 0 else num_threads
    
    filename = f"stochtree_bcf_python_n_{n}_p_{p}_num_gfr_{num_gfr}_num_mcmc_{num_mcmc}_dgp_num_{dgp_num}_snr_{snr_rounded}_test_set_pct_{test_set_pct_rounded}_num_threads_{num_threads_clean}.csv"
    output_dir = "tools/regression/bcf/stochtree_bcf_python_results"
    filename_full = os.path.join(output_dir, filename)
    results_df.to_csv(filename_full, index=False)
    print(f"Results saved to {filename_full}")


if __name__ == "__main__":
    main() 