import numpy as np
import pytest

from stochtree import BARTModel, BCFModel, OutcomeModel


def make_bart_data(n=100, p=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, p))
    y = np.sin(X[:, 0] * np.pi) + rng.normal(0, 0.1, n)
    n_train = int(0.8 * n)
    return X[:n_train], y[:n_train], X[n_train:], n_train, n - n_train


def make_bcf_data(n=100, p=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, p))
    pi_X = 0.25 + 0.5 * X[:, 0]
    Z = rng.binomial(1, pi_X, n).astype(float)
    y = pi_X * 5 + X[:, 1] * 2 * Z + rng.normal(0, 1, n)
    n_train = int(0.8 * n)
    return (
        X[:n_train], Z[:n_train], y[:n_train], pi_X[:n_train],
        X[n_train:], Z[n_train:], pi_X[n_train:],
        n_train, n - n_train,
    )


class TestBARTObservationWeights:
    def test_uniform_weights_match_no_weights(self):
        """Uniform weights of 1.0 should produce identical predictions to no weights."""
        X_train, y_train, X_test, n_train, n_test = make_bart_data()
        kwargs = dict(
            X_train=X_train, y_train=y_train, X_test=X_test,
            num_gfr=0, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": 1},
        )
        m1 = BARTModel()
        m1.sample(**kwargs)

        m2 = BARTModel()
        m2.sample(**kwargs, observation_weights=np.ones(n_train))

        np.testing.assert_array_equal(m1.y_hat_train, m2.y_hat_train)
        np.testing.assert_array_equal(m1.y_hat_test, m2.y_hat_test)

    def test_nonuniform_weights_output_shape(self):
        """Non-uniform weights: output shapes are correct."""
        X_train, y_train, X_test, n_train, n_test = make_bart_data()
        rng = np.random.default_rng(0)
        weights = rng.uniform(0.5, 2.0, n_train)
        num_mcmc = 10

        m = BARTModel()
        m.sample(
            X_train=X_train, y_train=y_train, X_test=X_test,
            observation_weights=weights,
            num_gfr=0, num_burnin=0, num_mcmc=num_mcmc,
        )
        assert m.y_hat_train.shape == (n_train, num_mcmc)
        assert m.y_hat_test.shape == (n_test, num_mcmc)

    def test_zero_weights_prior_mode(self):
        """All-zero weights with num_gfr=0 runs (prior sampling mode)."""
        X_train, y_train, _, n_train, _ = make_bart_data()
        num_mcmc = 10

        m = BARTModel()
        m.sample(
            X_train=X_train, y_train=y_train,
            observation_weights=np.zeros(n_train),
            num_gfr=0, num_burnin=0, num_mcmc=num_mcmc,
        )
        assert m.y_hat_train.shape == (n_train, num_mcmc)

    def test_invalid_type_raises(self):
        X_train, y_train, _, n_train, _ = make_bart_data()
        with pytest.raises(ValueError, match="numpy array"):
            BARTModel().sample(
                X_train=X_train, y_train=y_train,
                observation_weights=list(np.ones(n_train)),
                num_gfr=0, num_burnin=0, num_mcmc=5,
            )

    def test_2d_weights_raises(self):
        X_train, y_train, _, n_train, _ = make_bart_data()
        with pytest.raises(ValueError, match="1-dimensional"):
            BARTModel().sample(
                X_train=X_train, y_train=y_train,
                observation_weights=np.ones((n_train, 2)),
                num_gfr=0, num_burnin=0, num_mcmc=5,
            )

    def test_negative_weights_raises(self):
        X_train, y_train, _, n_train, _ = make_bart_data()
        weights = np.ones(n_train)
        weights[0] = -1.0
        with pytest.raises(ValueError, match="negative"):
            BARTModel().sample(
                X_train=X_train, y_train=y_train,
                observation_weights=weights,
                num_gfr=0, num_burnin=0, num_mcmc=5,
            )

    def test_all_zero_with_gfr_raises(self):
        X_train, y_train, _, n_train, _ = make_bart_data()
        with pytest.raises(ValueError, match="num_gfr"):
            BARTModel().sample(
                X_train=X_train, y_train=y_train,
                observation_weights=np.zeros(n_train),
                num_gfr=5, num_burnin=0, num_mcmc=10,
            )

    def test_cloglog_raises(self):
        rng = np.random.default_rng(0)
        n = 50
        X = rng.uniform(0, 1, (n, 3))
        y = rng.choice([1, 2, 3], n).astype(float)
        with pytest.raises(ValueError, match="cloglog"):
            BARTModel().sample(
                X_train=X, y_train=y,
                observation_weights=np.ones(n),
                num_gfr=0, num_burnin=0, num_mcmc=5,
                general_params={"outcome_model": OutcomeModel(outcome="ordinal", link="cloglog")},
            )

    def test_variance_forest_warns(self):
        X_train, y_train, _, n_train, _ = make_bart_data()
        with pytest.warns(UserWarning, match="variance forest"):
            BARTModel().sample(
                X_train=X_train, y_train=y_train,
                observation_weights=np.ones(n_train),
                num_gfr=0, num_burnin=0, num_mcmc=5,
                variance_forest_params={"num_trees": 5},
            )


class TestBCFObservationWeights:
    def test_uniform_weights_match_no_weights(self):
        """Uniform weights of 1.0 should produce identical predictions to no weights."""
        X_train, Z_train, y_train, pi_train, X_test, Z_test, pi_test, n_train, _ = make_bcf_data()
        kwargs = dict(
            X_train=X_train, Z_train=Z_train, y_train=y_train,
            propensity_train=pi_train, X_test=X_test, Z_test=Z_test, propensity_test=pi_test,
            num_gfr=0, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": 1},
        )
        m1 = BCFModel()
        m1.sample(**kwargs)

        m2 = BCFModel()
        m2.sample(**kwargs, observation_weights=np.ones(n_train))

        np.testing.assert_array_equal(m1.y_hat_train, m2.y_hat_train)
        np.testing.assert_array_equal(m1.tau_hat_train, m2.tau_hat_train)

    def test_nonuniform_weights_output_shape(self):
        X_train, Z_train, y_train, pi_train, X_test, Z_test, pi_test, n_train, n_test = make_bcf_data()
        rng = np.random.default_rng(0)
        weights = rng.uniform(0.5, 2.0, n_train)
        num_mcmc = 10

        m = BCFModel()
        m.sample(
            X_train=X_train, Z_train=Z_train, y_train=y_train,
            propensity_train=pi_train, X_test=X_test, Z_test=Z_test, propensity_test=pi_test,
            observation_weights=weights,
            num_gfr=0, num_burnin=0, num_mcmc=num_mcmc,
        )
        assert m.y_hat_train.shape == (n_train, num_mcmc)
        assert m.tau_hat_train.shape == (n_train, num_mcmc)
        assert m.y_hat_test.shape == (n_test, num_mcmc)
        assert m.tau_hat_test.shape == (n_test, num_mcmc)

    def test_negative_weights_raises(self):
        X_train, Z_train, y_train, pi_train, _, _, _, n_train, _ = make_bcf_data()
        weights = np.ones(n_train)
        weights[0] = -1.0
        with pytest.raises(ValueError, match="negative"):
            BCFModel().sample(
                X_train=X_train, Z_train=Z_train, y_train=y_train,
                propensity_train=pi_train,
                observation_weights=weights,
                num_gfr=0, num_burnin=0, num_mcmc=5,
            )

    def test_all_zero_with_gfr_raises(self):
        X_train, Z_train, y_train, pi_train, _, _, _, n_train, _ = make_bcf_data()
        with pytest.raises(ValueError, match="num_gfr"):
            BCFModel().sample(
                X_train=X_train, Z_train=Z_train, y_train=y_train,
                propensity_train=pi_train,
                observation_weights=np.zeros(n_train),
                num_gfr=5, num_burnin=0, num_mcmc=10,
            )

    def test_cloglog_raises(self):
        X_train, Z_train, y_train, pi_train, _, _, _, n_train, _ = make_bcf_data()
        with pytest.raises(ValueError, match="cloglog"):
            BCFModel().sample(
                X_train=X_train, Z_train=Z_train, y_train=y_train,
                propensity_train=pi_train,
                observation_weights=np.ones(n_train),
                num_gfr=0, num_burnin=0, num_mcmc=5,
                general_params={"outcome_model": OutcomeModel(outcome="binary", link="cloglog")},
            )
