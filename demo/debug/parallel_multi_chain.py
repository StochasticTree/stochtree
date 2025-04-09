# Multi Chain Demo Script

# Load necessary libraries
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from stochtree import BARTModel


def fit_bart(
    model_string,
    X_train,
    y_train,
    basis_train,
    X_test,
    basis_test,
    num_mcmc,
    gen_param_list,
    mean_list,
    i,
):
    bart_model = BARTModel()
    bart_model.sample(
        X_train=X_train,
        y_train=y_train,
        leaf_basis_train=basis_train,
        X_test=X_test,
        leaf_basis_test=basis_test,
        num_gfr=0,
        num_mcmc=num_mcmc,
        previous_model_json=model_string,
        previous_model_warmstart_sample_num=i,
        general_params=gen_param_list,
        mean_forest_params=mean_list,
    )
    return (bart_model.to_json(), bart_model.y_hat_test)


def bart_warmstart_parallel(X_train, y_train, basis_train, X_test, basis_test):
    # Run the GFR algorithm for a small number of iterations
    general_model_params = {"random_seed": -1}
    mean_forest_model_params = {"num_trees": 100}
    num_warmstart = 10
    num_mcmc = 100
    bart_model = BARTModel()
    bart_model.sample(
        X_train=X_train,
        y_train=y_train,
        leaf_basis_train=basis_train,
        X_test=X_test,
        leaf_basis_test=basis_test,
        num_gfr=num_warmstart,
        num_mcmc=0,
        general_params=general_model_params,
        mean_forest_params=mean_forest_model_params,
    )
    bart_model_json = bart_model.to_json()

    # Warm-start multiple BART fits from a different GFR forest
    process_tasks = [
        (
            bart_model_json,
            X_train,
            y_train,
            basis_train,
            X_test,
            basis_test,
            num_mcmc,
            general_model_params,
            mean_forest_model_params,
            i,
        )
        for i in range(4)
    ]
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(fit_bart, process_tasks)

    # Extract separate outputs as separate lists
    bart_model_json_list, bart_model_pred_list = zip(*results)

    # Process results
    combined_bart_model = BARTModel()
    combined_bart_model.from_json_string_list(bart_model_json_list)
    combined_bart_preds = bart_model_pred_list[0]
    for i in range(1, len(bart_model_pred_list)):
        combined_bart_preds = np.concatenate(
            (combined_bart_preds, bart_model_pred_list[i]), axis=1
        )

    return (combined_bart_model, combined_bart_preds)


if __name__ == "__main__":
    # RNG
    random_seed = 1234
    rng = np.random.default_rng(random_seed)

    # Generate covariates and basis
    n = 1000
    p_X = 10
    p_W = 1
    X = rng.uniform(0, 1, (n, p_X))
    W = rng.uniform(0, 1, (n, p_W))

    # Define the outcome mean function
    def outcome_mean(X, W):
        return np.where(
            (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
            -7.5 * W[:, 0],
            np.where(
                (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                -2.5 * W[:, 0],
                np.where(
                    (X[:, 0] >= 0.5) & (X[:, 0] < 0.75), 2.5 * W[:, 0], 7.5 * W[:, 0]
                ),
            ),
        )

    # Generate outcome
    f_XW = outcome_mean(X, W)
    epsilon = rng.normal(0, 1, n)
    y = f_XW + epsilon

    # Test-train split
    sample_inds = np.arange(n)
    train_inds, test_inds = train_test_split(
        sample_inds, test_size=0.2, random_state=random_seed
    )
    X_train = X[train_inds, :]
    X_test = X[test_inds, :]
    basis_train = W[train_inds, :]
    basis_test = W[test_inds, :]
    y_train = y[train_inds]
    y_test = y[test_inds]

    # Run the parallel BART
    combined_bart, combined_bart_preds = bart_warmstart_parallel(
        X_train, y_train, basis_train, X_test, basis_test
    )

    # Inspect the model outputs
    y_hat_mcmc = combined_bart.predict(X_test, basis_test)
    y_avg_mcmc = np.squeeze(y_hat_mcmc).mean(axis=1, keepdims=True)
    y_df = pd.DataFrame(
        np.concatenate((y_avg_mcmc, np.expand_dims(y_test, axis=1)), axis=1),
        columns=["Average BART Predictions", "Outcome"],
    )

    # Compare first warm-start chain to outcome
    sns.scatterplot(data=y_df, x="Average BART Predictions", y="Outcome")
    plt.axline((0, 0), slope=1, color="black", linestyle=(0, (3, 3)))
    plt.show()

    # Compare cached predictions to deserialized predictions for first chain
    chain_index = 0
    num_mcmc = 100
    offset_index = num_mcmc * chain_index
    chain_inds = slice(offset_index, (offset_index + num_mcmc))
    chain_1_preds_original = np.squeeze(combined_bart_preds[chain_inds]).mean(
        axis=1, keepdims=True
    )
    chain_1_preds_reloaded = np.squeeze(y_hat_mcmc[chain_inds]).mean(
        axis=1, keepdims=True
    )
    chain_df = pd.DataFrame(
        np.concatenate((chain_1_preds_reloaded, chain_1_preds_original), axis=1),
        columns=["New Predictions", "Original Predictions"],
    )
    sns.scatterplot(data=chain_df, x="New Predictions", y="Original Predictions")
    plt.axline((0, 0), slope=1, color="black", linestyle=(0, (3, 3)))
    plt.show()
