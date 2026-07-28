import itertools
import pytest
import numpy as np
from sklearn.model_selection import train_test_split

from stochtree import BARTModel, BCFModel, OutcomeModel


def run_bart_factorial(
    bart_data_train,
    bart_data_test,
    leaf_reg="none",
    variance_forest=False,
    random_effects="none",
    sampling_global_error_scale=False,
    sampling_leaf_scale=False,
    outcome_type="continuous",
    num_chains=1,
):
    # Unpack BART training data
    y = bart_data_train["y"]
    X = bart_data_train["X"]
    if leaf_reg != "none":
        leaf_basis = bart_data_train["leaf_basis"]
    else:
        leaf_basis = None
    if random_effects != "none":
        rfx_group_ids = bart_data_train["rfx_group_ids"]
    else:
        rfx_group_ids = None
    if random_effects == "custom":
        rfx_basis = bart_data_train["rfx_basis"]
    else:
        rfx_basis = None

    # Set BART model parameters
    general_params = {
        "num_chains": num_chains,
        "sample_sigma2_global": sampling_global_error_scale,
        "outcome_model": OutcomeModel(outcome=outcome_type),
    }
    mean_forest_params = {"sample_sigma2_leaf": sampling_leaf_scale}
    variance_forest_params = {"num_trees": 20 if variance_forest else 0}
    rfx_params = {
        "model_spec": "custom" if random_effects == "none" else random_effects
    }

    # Sample BART model
    bart_model = BARTModel()
    bart_model.sample(
        X_train=X,
        y_train=y,
        leaf_basis_train=leaf_basis,
        rfx_group_ids_train=rfx_group_ids,
        rfx_basis_train=rfx_basis,
        general_params=general_params,
        mean_forest_params=mean_forest_params,
        variance_forest_params=variance_forest_params,
        random_effects_params=rfx_params,
    )

    # Unpack test set data
    y_test = bart_data_test["y"]
    X_test = bart_data_test["X"]
    if leaf_reg != "none":
        leaf_basis_test = bart_data_test["leaf_basis"]
    else:
        leaf_basis_test = None
    if random_effects != "none":
        rfx_group_ids_test = bart_data_test["rfx_group_ids"]
    else:
        rfx_group_ids_test = None
    if random_effects == "custom":
        rfx_basis_test = bart_data_test["rfx_basis"]
    else:
        rfx_basis_test = None

    # Predict on test set
    mean_preds = bart_model.predict(
        X=X_test,
        leaf_basis=leaf_basis_test,
        rfx_group_ids=rfx_group_ids_test,
        rfx_basis=rfx_basis_test,
        type="mean",
        terms="all",
        scale="probability" if outcome_type == "binary" else "linear",
    )
    posterior_preds = bart_model.predict(
        X=X_test,
        leaf_basis=leaf_basis_test,
        rfx_group_ids=rfx_group_ids_test,
        rfx_basis=rfx_basis_test,
        type="posterior",
        terms="all",
        scale="probability" if outcome_type == "binary" else "linear",
    )

    # Compute intervals
    posterior_interval = bart_model.compute_posterior_interval(
        terms="all",
        level=0.95,
        scale="probability" if outcome_type == "binary" else "linear",
        X=X_test,
        leaf_basis=leaf_basis_test,
        rfx_group_ids=rfx_group_ids_test,
        rfx_basis=rfx_basis_test,
    )

    # Sample posterior predictive
    posterior_predictive_draws = bart_model.sample_posterior_predictive(
        X=X_test,
        leaf_basis=leaf_basis_test,
        rfx_group_ids=rfx_group_ids_test,
        rfx_basis=rfx_basis_test,
        num_draws_per_sample=5,
    )


def run_bcf_factorial(
    bcf_data_train,
    bcf_data_test,
    treatment_type="binary",
    variance_forest=False,
    random_effects="none",
    sampling_global_error_scale=False,
    sampling_mu_leaf_scale=False,
    sampling_tau_leaf_scale=False,
    outcome_type="continuous",
    num_chains=1,
    adaptive_coding=False,
    include_propensity=False,
):
    # Unpack BART training data
    y = bcf_data_train["y"]
    X = bcf_data_train["X"]
    Z = bcf_data_train["Z"]
    if include_propensity:
        propensity = bcf_data_train["propensity"]
    else:
        propensity = None
    if random_effects != "none":
        rfx_group_ids = bcf_data_train["rfx_group_ids"]
    else:
        rfx_group_ids = None
    if random_effects == "custom":
        rfx_basis = bcf_data_train["rfx_basis"]
    else:
        rfx_basis = None

    # Set BART model parameters
    general_params = {
        "num_chains": num_chains,
        "sample_sigma2_global": sampling_global_error_scale,
        "outcome_model": OutcomeModel(outcome=outcome_type),
        "adaptive_coding": adaptive_coding,
    }
    mu_forest_params = {"sample_sigma2_leaf": sampling_mu_leaf_scale}
    tau_forest_params = {"sample_sigma2_leaf": sampling_tau_leaf_scale}
    variance_forest_params = {"num_trees": 20 if variance_forest else 0}
    rfx_params = {
        "model_spec": "custom" if random_effects == "none" else random_effects
    }

    # Sample BART model
    bcf_model = BCFModel()
    bcf_model.sample(
        X_train=X,
        y_train=y,
        Z_train=Z,
        propensity_train=propensity,
        rfx_group_ids_train=rfx_group_ids,
        rfx_basis_train=rfx_basis,
        general_params=general_params,
        prognostic_forest_params=mu_forest_params,
        treatment_effect_forest_params=tau_forest_params,
        variance_forest_params=variance_forest_params,
        random_effects_params=rfx_params,
    )

    # Unpack test set data
    y_test = bcf_data_test["y"]
    X_test = bcf_data_test["X"]
    Z_test = bcf_data_test["Z"]
    if include_propensity:
        propensity_test = bcf_data_test["propensity"]
    else:
        propensity_test = None
    if random_effects != "none":
        rfx_group_ids_test = bcf_data_test["rfx_group_ids"]
    else:
        rfx_group_ids_test = None
    if random_effects == "custom":
        rfx_basis_test = bcf_data_test["rfx_basis"]
    else:
        rfx_basis_test = None

    # Predict on test set
    mean_preds = bcf_model.predict(
        X=X_test,
        Z=Z_test,
        propensity=propensity_test,
        rfx_group_ids=rfx_group_ids_test,
        rfx_basis=rfx_basis_test,
        type="mean",
        terms="all",
        scale="probability" if outcome_type == "binary" else "linear",
    )
    posterior_preds = bcf_model.predict(
        X=X_test,
        Z=Z_test,
        propensity=propensity_test,
        rfx_group_ids=rfx_group_ids_test,
        rfx_basis=rfx_basis_test,
        type="posterior",
        terms="all",
        scale="probability" if outcome_type == "binary" else "linear",
    )

    # Compute intervals
    posterior_interval = bcf_model.compute_posterior_interval(
        terms="all",
        level=0.95,
        scale="probability" if outcome_type == "binary" else "linear",
        X=X_test,
        Z=Z_test,
        propensity=propensity_test,
        rfx_group_ids=rfx_group_ids_test,
        rfx_basis=rfx_basis_test,
    )

    # Sample posterior predictive
    posterior_predictive_draws = bcf_model.sample_posterior_predictive(
        X=X_test,
        Z=Z_test,
        propensity=propensity_test,
        rfx_group_ids=rfx_group_ids_test,
        rfx_basis=rfx_basis_test,
        num_draws_per_sample=5,
    )


class TestAPICombinations:
    @pytest.mark.slow
    def test_bart_api_combinations(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Overall, we have seven components of a BART sampler which can be on / off or set to different levels:
        # 1. Leaf regression: none, univariate, multivariate
        # 2. Variance forest: no, yes
        # 3. Random effects: no, custom basis, `intercept_only`
        # 4. Sampling global error scale: no, yes
        # 5. Sampling leaf scale on mean forest: no, yes (only available for constant leaf or univariate leaf regression)
        # 6. Outcome type: continuous (identity link), binary (probit link)
        # 7. Number of chains: 1, >1
        #
        # For each of the possible models this implies,
        # we'd like to be sure that stochtree functions that operate on BART models
        # will run without error. Since there are so many possible models implied by the
        # options above, this test is designed to be quick (small sample size, low dimensional data)
        # and we are only interested in ensuring no errors are triggered.

        # Generate data with random effects
        n = 50
        p = 3
        num_basis = 2
        num_rfx_groups = 3
        num_rfx_basis = 2
        X = rng.uniform(0, 1, (n, p))
        leaf_basis = rng.uniform(0, 1, (n, num_basis))
        leaf_coefs = rng.uniform(0, 1, num_basis)
        group_ids = rng.choice(num_rfx_groups, size=n)
        rfx_basis = rng.uniform(0, 1, (n, num_rfx_basis))
        rfx_coefs = rng.uniform(0, 1, (num_rfx_groups, num_rfx_basis))
        mean_term = np.sin(X[:, 0]) * np.sum(leaf_basis * leaf_coefs, axis=1)
        rfx_term = np.sum(rfx_coefs[group_ids - 1, :] * rfx_basis, axis=1)
        E_y = mean_term + rfx_term
        E_y = E_y - np.mean(E_y)
        epsilon = rng.normal(0, 1, n)
        y_continuous = E_y + epsilon
        y_binary = (y_continuous > 0).astype(int)

        # Split into test and train sets
        test_set_pct = 0.5
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=test_set_pct)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        leaf_basis_train = leaf_basis[train_inds, :]
        leaf_basis_test = leaf_basis[test_inds, :]
        rfx_basis_train = rfx_basis[train_inds, :]
        rfx_basis_test = rfx_basis[test_inds, :]
        group_ids_train = group_ids[train_inds]
        group_ids_test = group_ids[test_inds]
        y_continuous_train = y_continuous[train_inds]
        y_continuous_test = y_continuous[test_inds]
        y_binary_train = y_binary[train_inds]
        y_binary_test = y_binary[test_inds]

        # Run the power set of models
        leaf_reg_options = ["none", "univariate", "multivariate"]
        variance_forest_options = [False, True]
        random_effects_options = ["none", "custom", "intercept_only"]
        sampling_global_error_scale_options = [False, True]
        sampling_leaf_scale_options = [False, True]
        outcome_type_options = ["continuous", "binary"]
        num_chains_options = [1, 3]
        model_options_iter = itertools.product(
            leaf_reg_options,
            variance_forest_options,
            random_effects_options,
            sampling_global_error_scale_options,
            sampling_leaf_scale_options,
            outcome_type_options,
            num_chains_options,
        )
        for i, options in enumerate(model_options_iter):
            # Unpack BART train and test data
            bart_data_train = {}
            bart_data_test = {}
            bart_data_train["X"] = X_train
            bart_data_test["X"] = X_test
            if options[5] == "continuous":
                bart_data_train["y"] = y_continuous_train
                bart_data_test["y"] = y_continuous_test
            else:
                bart_data_train["y"] = y_binary_train
                bart_data_test["y"] = y_binary_test
            if options[0] != "none":
                if options[0] == "univariate":
                    bart_data_train["leaf_basis"] = leaf_basis_train[:, 0]
                    bart_data_test["leaf_basis"] = leaf_basis_test[:, 0]
                else:
                    bart_data_train["leaf_basis"] = leaf_basis_train
                    bart_data_test["leaf_basis"] = leaf_basis_test
            else:
                bart_data_train["leaf_basis"] = None
                bart_data_test["leaf_basis"] = None
            if options[2] != "none":
                bart_data_train["rfx_group_ids"] = group_ids_train
                bart_data_test["rfx_group_ids"] = group_ids_test
            else:
                bart_data_train["rfx_group_ids"] = None
                bart_data_test["rfx_group_ids"] = None
            if options[2] == "custom":
                bart_data_train["rfx_basis"] = rfx_basis_train
                bart_data_test["rfx_basis"] = rfx_basis_test
            else:
                bart_data_train["rfx_basis"] = None
                bart_data_test["rfx_basis"] = None

            # Determine whether this combination should throw an error, raise a warning, or run as intended
            error_cond = (options[1]) and (options[5] == "binary")
            warning_cond_1 = (options[4]) and (options[0] == "multivariate")
            warning_cond_2 = (options[3]) and (options[5] == "binary")
            warning_cond_3 = (options[3]) and (options[1])
            warning_cond = warning_cond_1 or warning_cond_2 or warning_cond_3
            if error_cond and warning_cond:
                with pytest.raises(ValueError) as excinfo:
                    with pytest.warns(UserWarning) as warninfo:
                        run_bart_factorial(
                            bart_data_train=bart_data_train,
                            bart_data_test=bart_data_test,
                            leaf_reg=options[0],
                            variance_forest=options[1],
                            random_effects=options[2],
                            sampling_global_error_scale=options[3],
                            sampling_leaf_scale=options[4],
                            outcome_type=options[5],
                            num_chains=options[6],
                        )
            elif error_cond and not warning_cond:
                with pytest.raises(ValueError) as excinfo:
                    run_bart_factorial(
                        bart_data_train=bart_data_train,
                        bart_data_test=bart_data_test,
                        leaf_reg=options[0],
                        variance_forest=options[1],
                        random_effects=options[2],
                        sampling_global_error_scale=options[3],
                        sampling_leaf_scale=options[4],
                        outcome_type=options[5],
                        num_chains=options[6],
                    )
            elif not error_cond and warning_cond:
                with pytest.warns(UserWarning) as warninfo:
                    run_bart_factorial(
                        bart_data_train=bart_data_train,
                        bart_data_test=bart_data_test,
                        leaf_reg=options[0],
                        variance_forest=options[1],
                        random_effects=options[2],
                        sampling_global_error_scale=options[3],
                        sampling_leaf_scale=options[4],
                        outcome_type=options[5],
                        num_chains=options[6],
                    )
            else:
                run_bart_factorial(
                    bart_data_train=bart_data_train,
                    bart_data_test=bart_data_test,
                    leaf_reg=options[0],
                    variance_forest=options[1],
                    random_effects=options[2],
                    sampling_global_error_scale=options[3],
                    sampling_leaf_scale=options[4],
                    outcome_type=options[5],
                    num_chains=options[6],
                )

    @pytest.mark.slow
    def test_bcf_api_combinations(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Overall, we have nine components of a BCF sampler which can be on / off or set to different levels:
        # 1. treatment: binary, univariate continuous, multivariate
        # 2. Variance forest: no, yes
        # 3. Random effects: no, custom basis, `intercept_only`, `intercept_plus_treatment`
        # 4. Sampling global error scale: no, yes
        # 5. Sampling leaf scale on prognostic forest: no, yes
        # 6. Sampling leaf scale on treatment forest: no, yes (only available for univariate treatment)
        # 7. Outcome type: continuous (identity link), binary (probit link)
        # 8. Number of chains: 1, >1
        # 9. Adaptive coding: no, yes
        #
        # For each of the possible models this implies,
        # we'd like to be sure that stochtree functions that operate on BCF models
        # will run without error. Since there are so many possible models implied by the
        # options above, this test is designed to be quick (small sample size, low dimensional data)
        # and we are only interested in ensuring no errors are triggered.

        # Generate data with random effects
        n = 50
        p = 3
        num_rfx_groups = 3
        num_rfx_basis = 2
        X = rng.uniform(0, 1, (n, p))
        binary_treatment = rng.binomial(1, 0.5, n)
        continuous_treatment = rng.uniform(0, 1, n)
        multivariate_treatment = np.column_stack(
            (binary_treatment, continuous_treatment)
        )
        propensity = rng.uniform(0, 1, n)
        group_ids = rng.choice(num_rfx_groups, size=n)
        rfx_basis = rng.uniform(0, 1, (n, num_rfx_basis))
        rfx_coefs = rng.uniform(0, 1, (num_rfx_groups, num_rfx_basis))
        prognostic_term = np.sin(X[:, 0])
        binary_treatment_effect = X[:, 1]
        continuous_treatment_effect = X[:, 2]
        rfx_term = np.sum(rfx_coefs[group_ids - 1, :] * rfx_basis, axis=1)
        E_y = (prognostic_term +
            binary_treatment_effect * binary_treatment +
            continuous_treatment_effect * continuous_treatment +
            rfx_term)
        E_y = E_y - np.mean(E_y)
        epsilon = rng.normal(0, 1, n)
        y_continuous = E_y + epsilon
        y_binary = (y_continuous > 0).astype(int)

        # Split into test and train sets
        test_set_pct = 0.5
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=test_set_pct)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        binary_treatment_test = binary_treatment[test_inds]
        binary_treatment_train = binary_treatment[train_inds]
        propensity_test = propensity[test_inds]
        propensity_train = propensity[train_inds]
        continuous_treatment_test = continuous_treatment[test_inds]
        continuous_treatment_train = continuous_treatment[train_inds]
        multivariate_treatment_test = multivariate_treatment[test_inds, ]
        multivariate_treatment_train = multivariate_treatment[train_inds, ]
        rfx_basis_train = rfx_basis[train_inds, :]
        rfx_basis_test = rfx_basis[test_inds, :]
        group_ids_train = group_ids[train_inds]
        group_ids_test = group_ids[test_inds]
        y_continuous_train = y_continuous[train_inds]
        y_continuous_test = y_continuous[test_inds]
        y_binary_train = y_binary[train_inds]
        y_binary_test = y_binary[test_inds]

        # Run the power set of models
        treatment_options = ["binary", "univariate_continuous", "multivariate"]
        variance_forest_options = [False, True]
        random_effects_options = ["none", "custom", "intercept_only", "intercept_plus_treatment"]
        sampling_global_error_scale_options = [False, True]
        sampling_mu_leaf_scale_options = [False, True]
        sampling_tau_leaf_scale_options = [False, True]
        outcome_type_options = ["continuous", "binary"]
        num_chains_options = [1, 3]
        adaptive_coding_options = [False, True]
        include_propensity_options = [False, True]
        model_options_iter = itertools.product(
            treatment_options,
            variance_forest_options,
            random_effects_options,
            sampling_global_error_scale_options,
            sampling_mu_leaf_scale_options,
            sampling_tau_leaf_scale_options,
            outcome_type_options,
            num_chains_options,
            adaptive_coding_options, 
            include_propensity_options
        )
        for i, options in enumerate(model_options_iter):
            # Unpack BCF train and test data
            bcf_data_train = {}
            bcf_data_test = {}
            bcf_data_train["X"] = X_train
            bcf_data_test["X"] = X_test
            bcf_data_train["propensity"] = propensity_train
            bcf_data_test["propensity"] = propensity_test
            if options[5] == "continuous":
                bcf_data_train["y"] = y_continuous_train
                bcf_data_test["y"] = y_continuous_test
            else:
                bcf_data_train["y"] = y_binary_train
                bcf_data_test["y"] = y_binary_test
            if options[0] == "binary":
                bcf_data_train["Z"] = binary_treatment_train
                bcf_data_test["Z"] = binary_treatment_test
            elif options[0] == "univariate_continuous":
                bcf_data_train["Z"] = continuous_treatment_train
                bcf_data_test["Z"] = continuous_treatment_test
            else:
                bcf_data_train["Z"] = multivariate_treatment_train
                bcf_data_test["Z"] = multivariate_treatment_test
            if options[2] != "none":
                bcf_data_train["rfx_group_ids"] = group_ids_train
                bcf_data_test["rfx_group_ids"] = group_ids_test
            else:
                bcf_data_train["rfx_group_ids"] = None
                bcf_data_test["rfx_group_ids"] = None
            if options[2] == "custom":
                bcf_data_train["rfx_basis"] = rfx_basis_train
                bcf_data_test["rfx_basis"] = rfx_basis_test
            else:
                bcf_data_train["rfx_basis"] = None
                bcf_data_test["rfx_basis"] = None

            # Determine whether this combination should throw an error, raise a warning, or run as intended
            error_cond = (options[1]) and (options[6] == "binary")
            warning_cond_1 = (options[5]) and (options[0] == "multivariate")
            warning_cond_2 = (options[3]) and (options[6] == "binary")
            warning_cond_3 = (options[3]) and (options[1])
            warning_cond_4 = (options[8]) and (options[0] != "binary")
            warning_cond_5 = (not options[9]) and (options[0] == "multivariate")
            warning_cond_6 = (options[2] == "intercept_plus_treatment") and (options[0] == "multivariate")
            warning_cond = warning_cond_1 or warning_cond_2 or warning_cond_3 or warning_cond_4 or warning_cond_5 or warning_cond_6
            print(f"error_cond: {error_cond}, warning_cond_1: {warning_cond_1}, warning_cond_2: {warning_cond_2}, warning_cond_3: {warning_cond_3}, warning_cond_4: {warning_cond_4}, warning_cond_5: {warning_cond_5}, warning_cond_6: {warning_cond_6}")
            if error_cond and warning_cond:
                with pytest.raises(ValueError) as excinfo:
                    with pytest.warns(UserWarning) as warninfo:
                        run_bcf_factorial(
                            bcf_data_train=bcf_data_train,
                            bcf_data_test=bcf_data_test,
                            treatment_type=options[0],
                            variance_forest=options[1],
                            random_effects=options[2],
                            sampling_global_error_scale=options[3],
                            sampling_mu_leaf_scale=options[4],
                            sampling_tau_leaf_scale=options[5],
                            outcome_type=options[6],
                            num_chains=options[7],
                            adaptive_coding=options[8],
                            include_propensity=options[9],
                        )
            elif error_cond and not warning_cond:
                with pytest.raises(ValueError) as excinfo:
                    run_bcf_factorial(
                        bcf_data_train=bcf_data_train,
                        bcf_data_test=bcf_data_test,
                        treatment_type=options[0],
                        variance_forest=options[1],
                        random_effects=options[2],
                        sampling_global_error_scale=options[3],
                        sampling_mu_leaf_scale=options[4],
                        sampling_tau_leaf_scale=options[5],
                        outcome_type=options[6],
                        num_chains=options[7],
                        adaptive_coding=options[8],
                        include_propensity=options[9],
                    )
            elif not error_cond and warning_cond:
                with pytest.warns(UserWarning) as warninfo:
                    run_bcf_factorial(
                        bcf_data_train=bcf_data_train,
                        bcf_data_test=bcf_data_test,
                        treatment_type=options[0],
                        variance_forest=options[1],
                        random_effects=options[2],
                        sampling_global_error_scale=options[3],
                        sampling_mu_leaf_scale=options[4],
                        sampling_tau_leaf_scale=options[5],
                        outcome_type=options[6],
                        num_chains=options[7],
                        adaptive_coding=options[8],
                        include_propensity=options[9],
                    )
            else:
                run_bcf_factorial(
                    bcf_data_train=bcf_data_train,
                    bcf_data_test=bcf_data_test,
                    treatment_type=options[0],
                    variance_forest=options[1],
                    random_effects=options[2],
                    sampling_global_error_scale=options[3],
                    sampling_mu_leaf_scale=options[4],
                    sampling_tau_leaf_scale=options[5],
                    outcome_type=options[6],
                    num_chains=options[7],
                    adaptive_coding=options[8],
                    include_propensity=options[9],
                )
