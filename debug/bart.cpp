/*!
 * Standalone BART driver for smoke-testing and profiling.
 *
 * Supports all models exposed by the C++ BARTFit dispatch API:
 *   identity      — continuous outcome, standard BART
 *   probit        — binary outcome, Albert-Chib augmentation
 *   varforest     — variance-forest only, heteroskedastic DGP
 *   cloglog       — binary cloglog (K=2), OrdinalSampler augmentation
 *   ordinal       — ordinal cloglog (K=3)
 *   mean+varforest— continuous mean forest + variance forest
 *   leaf-reg      — univariate leaf regression (linear leaf, 1-column basis)
 *   leaf-reg-mv   — multivariate leaf regression (2-column basis)
 *   rfx           — intercept-only random effects (10 groups)
 *
 * Run a single model:
 *   ./build/debug_bart --model identity
 *   ./build/debug_bart --model rfx
 *
 * Run all smoke tests (default):
 *   ./build/debug_bart
 *
 * Run all smoke tests + wall-time table:
 *   ./build/debug_bart --timing
 *
 * Build (debugging — unoptimized, full symbols):
 *   cmake -DBUILD_DEBUG_TARGETS=ON -DCMAKE_BUILD_TYPE=Debug \
 *         -B build && cmake --build build --target debug_bart
 *
 * Build (profiling — optimized with symbols):
 *   cmake -DBUILD_DEBUG_TARGETS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
 *         -B build && cmake --build build --target debug_bart
 *
 * See debug/README.md for profiling instructions (macOS Instruments, Linux perf).
 */

#include <stochtree/bart.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double elapsed_ms(Clock::time_point t0) {
    return Ms(Clock::now() - t0).count();
}

// ── Helpers ─────────────────────────────────────────────────────────────────

// Phi(x) = 0.5 * erfc(-x/sqrt(2))
static double Phi(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

static bool all_finite(const std::vector<double>& v) {
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
}

// ── Synthetic data generators ────────────────────────────────────────────────

// Continuous DGP: y = 2*x0 - x1 + N(0, 0.5^2).  Covariates ~ N(0,1).
static void make_continuous_data(int n_train, int n_test, int p,
                                  std::vector<double>& X_train,
                                  std::vector<double>& y_train,
                                  std::vector<double>& X_test,
                                  unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    X_train.resize(static_cast<size_t>(n_train) * p);
    y_train.resize(n_train);
    X_test.resize(static_cast<size_t>(n_test) * p);

    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < n_train; ++i)
            X_train[j * n_train + i] = normal(rng);
        for (int i = 0; i < n_test; ++i)
            X_test[j * n_test + i] = normal(rng);
    }
    for (int i = 0; i < n_train; ++i)
        y_train[i] = 2.0 * X_train[i] - X_train[n_train + i] + 0.5 * normal(rng);
}

// Binary DGP for probit: eta = x0 - 0.5*x1, P(y=1|x) = Phi(eta).
static void make_binary_data(int n, int p,
                              std::vector<double>& X,
                              std::vector<double>& y,
                              unsigned seed = 123)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    X.resize(static_cast<size_t>(n) * p);
    y.resize(n);

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < n; ++i)
            X[j * n + i] = normal(rng);

    for (int i = 0; i < n; ++i) {
        double eta = X[i];
        if (p > 1) eta -= 0.5 * X[n + i];
        y[i] = (unif(rng) < Phi(eta)) ? 1.0 : 0.0;
    }
}

// Ordinal cloglog DGP: K categories; eta = x0 - 0.5*x1.
// Cutpoints gamma_k uniformly spaced in (-2, 2).
// P(Y <= k | x) = 1 - exp(-exp(gamma_k + eta)).
static void make_ordinal_data(int n, int p, int K,
                               std::vector<double>& X,
                               std::vector<double>& y,
                               unsigned seed = 99)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    X.resize(static_cast<size_t>(n) * p);
    y.resize(n);

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < n; ++i)
            X[j * n + i] = normal(rng);

    // Cutpoints uniformly spaced in (-2, 2)
    std::vector<double> cuts(K - 1);
    for (int k = 0; k < K - 1; ++k)
        cuts[k] = -2.0 + 4.0 * (k + 1.0) / K;

    for (int i = 0; i < n; ++i) {
        double eta = X[i] - 0.5 * (p > 1 ? X[n + i] : 0.0);
        double u   = unif(rng);
        y[i] = static_cast<double>(K - 1);
        for (int k = 0; k < K - 1; ++k) {
            double pcdf = 1.0 - std::exp(-std::exp(cuts[k] + eta));
            if (u < pcdf) { y[i] = static_cast<double>(k); break; }
        }
    }
}

// Heteroskedastic DGP: zero mean, variance driven by a partition on x0.
// True conditional std: s(x) = {0.5, 1.0, 2.0, 3.0} based on quartile of x0.
static void make_heterosked_data(int n, int p,
                                  std::vector<double>& X,
                                  std::vector<double>& y,
                                  std::vector<double>& s_true,
                                  unsigned seed = 789)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    X.resize(static_cast<size_t>(n) * p);
    y.resize(n);
    s_true.resize(n);

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < n; ++i)
            X[j * n + i] = unif(rng);

    for (int i = 0; i < n; ++i) {
        double x0 = X[i];
        double s  = (x0 < 0.25) ? 0.5 : (x0 < 0.5) ? 1.0 : (x0 < 0.75) ? 2.0 : 3.0;
        s_true[i] = s;
        y[i] = s * normal(rng);
    }
}

// Combined mean + heteroskedastic DGP.
// Mean: 2*x0 - x1 (using columns 0,1 of X, normal).
// Std:  s(x2) = {0.5,1,2,3} based on quartile of x2 (uniform).
static void make_mean_varforest_data(int n, int p,
                                      std::vector<double>& X,
                                      std::vector<double>& y,
                                      std::vector<double>& s_true,
                                      unsigned seed = 55)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    X.resize(static_cast<size_t>(n) * p);
    y.resize(n);
    s_true.resize(n);

    // First two columns normal, rest uniform (for variance partition)
    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < n; ++i)
            X[j * n + i] = (j < 2) ? normal(rng) : unif(rng);
    }

    for (int i = 0; i < n; ++i) {
        double x0 = X[i];
        double x1 = (p > 1) ? X[n + i] : 0.0;
        double x2 = (p > 2) ? X[2 * n + i] : 0.5;
        double s  = (x2 < 0.25) ? 0.5 : (x2 < 0.5) ? 1.0 : (x2 < 0.75) ? 2.0 : 3.0;
        s_true[i] = s;
        y[i] = 2.0 * x0 - x1 + s * normal(rng);
    }
}

// Leaf regression DGP: y_i = x0_i * b0_i [+ x1_i * b1_i] + N(0, 0.3^2).
// X ~ N(0,1), basis ~ N(0,1), column-major.
static void make_leaf_reg_data(int n_train, int n_test, int p, int basis_dim,
                                std::vector<double>& X_train,
                                std::vector<double>& y_train,
                                std::vector<double>& X_test,
                                std::vector<double>& basis_train,
                                std::vector<double>& basis_test,
                                unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.3);

    X_train.resize(static_cast<size_t>(n_train) * p);
    X_test.resize(static_cast<size_t>(n_test) * p);
    basis_train.resize(static_cast<size_t>(n_train) * basis_dim);
    basis_test.resize(static_cast<size_t>(n_test) * basis_dim);
    y_train.resize(n_train);

    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < n_train; ++i)
            X_train[j * n_train + i] = normal(rng);
        for (int i = 0; i < n_test; ++i)
            X_test[j * n_test + i] = normal(rng);
    }
    for (int j = 0; j < basis_dim; ++j) {
        for (int i = 0; i < n_train; ++i)
            basis_train[j * n_train + i] = normal(rng);
        for (int i = 0; i < n_test; ++i)
            basis_test[j * n_test + i] = normal(rng);
    }

    // DGP: y_i = sum_j x_j(i) * basis_j(i) + noise
    for (int i = 0; i < n_train; ++i) {
        y_train[i] = noise(rng);
        for (int j = 0; j < basis_dim; ++j)
            y_train[i] += X_train[j * n_train + i] * basis_train[j * n_train + i];
    }
}

// Random effects DGP: y_i = 2*x0 - x1 + alpha_g[i] + N(0, 0.3^2).
// Group effects alpha_g ~ N(0, 0.5^2); groups = 0..num_groups-1 cycling.
static void make_rfx_data(int n, int p, int num_groups,
                           std::vector<double>& X,
                           std::vector<double>& y,
                           std::vector<int>& groups,
                           unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::normal_distribution<double> group_dist(0.0, 0.5);
    std::normal_distribution<double> eps(0.0, 0.3);

    X.resize(static_cast<size_t>(n) * p);
    y.resize(n);
    groups.resize(n);

    for (int j = 0; j < p; ++j)
        for (int i = 0; i < n; ++i)
            X[j * n + i] = normal(rng);

    std::vector<double> alpha(num_groups);
    for (int g = 0; g < num_groups; ++g)
        alpha[g] = group_dist(rng);

    for (int i = 0; i < n; ++i) {
        groups[i] = i % num_groups;
        double mean_i = 2.0 * X[i] - (p > 1 ? X[n + i] : 0.0);
        y[i] = mean_i + alpha[groups[i]] + eps(rng);
    }
}

// ── Summary statistics ───────────────────────────────────────────────────────

// Posterior-mean RMSE.  pred is n × S column-major (n obs, S samples).
static double rmse(const std::vector<double>& pred,
                   const std::vector<double>& truth,
                   int n, int S)
{
    std::vector<double> y_bar(n, 0.0);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < n; ++i)
            y_bar[i] += pred[static_cast<size_t>(s) * n + i];
    for (int i = 0; i < n; ++i) y_bar[i] /= S;

    double sse = 0.0;
    for (int i = 0; i < n; ++i) { double e = y_bar[i] - truth[i]; sse += e * e; }
    return std::sqrt(sse / n);
}

// ── 1. Identity BART smoke test ──────────────────────────────────────────────

static void run_identity_smoke_test()
{
    constexpr int n_train = 500, n_test = 100, p = 3;
    constexpr int num_trees = 50, num_gfr = 10, num_burnin = 50, num_mcmc = 100;

    std::vector<double> X_train, y_train, X_test;
    make_continuous_data(n_train, n_test, p, X_train, y_train, X_test);

    // True test outcomes (noiseless signal)
    std::vector<double> y_test(n_test);
    for (int i = 0; i < n_test; ++i)
        y_test[i] = 2.0 * X_test[i] - X_test[n_test + i];

    StochTree::BARTConfig config;
    config.num_trees            = num_trees;
    config.num_gfr              = num_gfr;
    config.num_burnin           = num_burnin;
    config.num_mcmc             = num_mcmc;
    config.keep_gfr             = false;
    config.keep_burnin          = false;
    config.sample_sigma2_global = true;
    config.sample_sigma2_leaf   = true;
    config.random_seed          = 42;

    StochTree::BARTData data;
    data.X_train = X_train.data(); data.n_train = n_train; data.p = p;
    data.y_train = y_train.data();
    data.X_test  = X_test.data();  data.n_test  = n_test;

    StochTree::BARTResult result;
    StochTree::BARTFit(&result, config, data);

    int S = result.num_total_samples;
    std::cout << "\n=== identity: Identity BART smoke test ===\n"
              << "  n_train=" << n_train << "  n_test=" << n_test
              << "  p=" << p << "  trees=" << num_trees
              << "  GFR=" << num_gfr << "  burnin=" << num_burnin
              << "  MCMC=" << num_mcmc << "  samples=" << S << "\n";
    std::cout << std::fixed << std::setprecision(4)
              << "  Train RMSE: " << rmse(result.y_hat_train, y_train, n_train, S) << "\n"
              << "  Test  RMSE: " << rmse(result.y_hat_test,  y_test,  n_test,  S) << "\n";
    if (!result.sigma2_global_samples.empty()) {
        double s2 = std::accumulate(result.sigma2_global_samples.begin(),
                                    result.sigma2_global_samples.end(), 0.0) / S;
        std::cout << "  sigma2_global posterior mean: " << s2 << "\n";
    }
    std::cout << "  ForestContainer samples: " << result.forest_container->NumSamples() << "\n";
}

// ── 2. Probit BART smoke test ────────────────────────────────────────────────

static void run_probit_smoke_test()
{
    constexpr int n_train = 500, n_test = 100, p = 3;
    constexpr int num_trees = 50, num_gfr = 10, num_burnin = 50, num_mcmc = 100;

    std::vector<double> X_train, y_train, X_test, y_test_unused;
    make_binary_data(n_train, p, X_train, y_train);
    make_binary_data(n_test,  p, X_test,  y_test_unused, /*seed=*/456);

    StochTree::BARTConfig config;
    config.num_trees            = num_trees;
    config.num_gfr              = num_gfr;
    config.num_burnin           = num_burnin;
    config.num_mcmc             = num_mcmc;
    config.keep_gfr             = false;
    config.keep_burnin          = false;
    config.link_function        = StochTree::LinkFunction::Probit;
    config.sample_sigma2_global = false;
    config.sample_sigma2_leaf   = true;
    config.random_seed          = 42;

    StochTree::BARTData data;
    data.X_train = X_train.data(); data.n_train = n_train; data.p = p;
    data.y_train = y_train.data();
    data.X_test  = X_test.data();  data.n_test  = n_test;

    StochTree::BARTResult result;
    StochTree::BARTFit(&result, config, data);

    int S = result.num_total_samples;

    int n_pos = 0;
    for (double v : y_train) if (v > 0.5) n_pos++;
    double y_bar_prob = static_cast<double>(n_pos) / n_train;

    std::vector<double> p_hat(n_train, 0.0);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < n_train; ++i)
            p_hat[i] += Phi(result.y_hat_train[static_cast<size_t>(s) * n_train + i]);
    for (int i = 0; i < n_train; ++i) p_hat[i] /= S;
    double mean_p_hat = std::accumulate(p_hat.begin(), p_hat.end(), 0.0) / n_train;

    std::cout << "\n=== probit: Probit BART smoke test ===\n"
              << "  n_train=" << n_train << "  n_test=" << n_test
              << "  p=" << p << "  trees=" << num_trees
              << "  GFR=" << num_gfr << "  burnin=" << num_burnin
              << "  MCMC=" << num_mcmc << "  samples=" << S << "\n";
    std::cout << std::fixed << std::setprecision(4)
              << "  Observed class rate:       " << y_bar_prob << "\n"
              << "  Mean Phi(latent) on train: " << mean_p_hat
              << "  (should be close to class rate)\n"
              << "  sigma2_global_samples empty: "
              << (result.sigma2_global_samples.empty() ? "yes" : "NO — BUG") << "\n"
              << "  leaf_scale samples: " << result.leaf_scale_samples.size() << "\n"
              << "  ForestContainer samples: " << result.forest_container->NumSamples() << "\n";
}

// ── 3. Variance-forest smoke test ────────────────────────────────────────────

static void run_varforest_smoke_test()
{
    constexpr int n_train = 500, n_test = 100, p = 5;
    constexpr int num_trees_var = 50, num_gfr = 10, num_burnin = 50, num_mcmc = 100;

    std::vector<double> X_train, y_train, s_true_train;
    std::vector<double> X_test,  y_test,  s_true_test;
    make_heterosked_data(n_train, p, X_train, y_train, s_true_train);
    make_heterosked_data(n_test,  p, X_test,  y_test,  s_true_test, 790);

    StochTree::BARTConfig config;
    config.num_trees              = 0;       // variance-only model
    config.num_gfr                = num_gfr;
    config.num_burnin             = num_burnin;
    config.num_mcmc               = num_mcmc;
    config.keep_gfr               = false;
    config.keep_burnin            = false;
    config.include_variance_forest = true;
    config.num_trees_variance      = num_trees_var;
    config.sample_sigma2_global    = false;
    config.sample_sigma2_leaf      = false;
    config.random_seed             = 42;

    StochTree::BARTData data;
    data.X_train = X_train.data(); data.n_train = n_train; data.p = p;
    data.y_train = y_train.data();
    data.X_test  = X_test.data();  data.n_test  = n_test;

    StochTree::BARTResult result;
    StochTree::BARTFit(&result, config, data);

    int S = result.num_total_samples;

    std::vector<double> s_hat(n_test, 0.0);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < n_test; ++i)
            s_hat[i] += std::sqrt(result.sigma2_x_hat_test[static_cast<size_t>(s) * n_test + i]);
    for (int i = 0; i < n_test; ++i) s_hat[i] /= S;

    double rmse_s = 0.0;
    for (int i = 0; i < n_test; ++i) { double e = s_hat[i] - s_true_test[i]; rmse_s += e * e; }
    rmse_s = std::sqrt(rmse_s / n_test);

    double s_bar = 0.0;
    for (double v : s_true_test) s_bar += v;
    s_bar /= n_test;
    double rmse_null = 0.0;
    for (int i = 0; i < n_test; ++i) { double e = s_bar - s_true_test[i]; rmse_null += e * e; }
    rmse_null = std::sqrt(rmse_null / n_test);

    std::cout << "\n=== varforest: Variance-forest smoke test ===\n"
              << "  n_train=" << n_train << "  n_test=" << n_test
              << "  p=" << p << "  var_trees=" << num_trees_var
              << "  GFR=" << num_gfr << "  burnin=" << num_burnin
              << "  MCMC=" << num_mcmc << "  samples=" << S << "\n";
    std::cout << std::fixed << std::setprecision(4)
              << "  Test RMSE for E[sqrt(sigma2_x)]: " << rmse_s
              << "  (null: " << rmse_null << ")\n"
              << "  ForestContainer (variance) samples: "
              << result.variance_forest_container->NumSamples() << "\n";
}

// ── 4. Cloglog / ordinal smoke test ──────────────────────────────────────────

static void run_cloglog_smoke_test(int K = 2)
{
    const char* tag = (K == 2) ? "cloglog" : "ordinal";
    constexpr int n_train = 400, n_test = 100, p = 5;
    constexpr int num_trees = 50, num_gfr = 5, num_burnin = 0, num_mcmc = 50;

    std::vector<double> X_train, y_train, X_test, y_test_unused;
    make_ordinal_data(n_train, p, K, X_train, y_train);
    make_ordinal_data(n_test,  p, K, X_test,  y_test_unused, /*seed=*/200);

    StochTree::BARTConfig config;
    config.num_trees               = num_trees;
    config.num_gfr                 = num_gfr;
    config.num_burnin              = num_burnin;
    config.num_mcmc                = num_mcmc;
    config.keep_gfr                = false;
    config.keep_burnin             = false;
    config.link_function           = StochTree::LinkFunction::Cloglog;
    config.cloglog_num_categories  = K;
    config.sample_sigma2_global    = false;
    config.sample_sigma2_leaf      = false;
    config.random_seed             = 42;

    StochTree::BARTData data;
    data.X_train = X_train.data(); data.n_train = n_train; data.p = p;
    data.y_train = y_train.data();
    data.X_test  = X_test.data();  data.n_test  = n_test;

    StochTree::BARTResult result;
    StochTree::BARTFit(&result, config, data);

    int S = result.num_total_samples;

    // Category distribution in training data
    std::vector<int> counts(K, 0);
    for (double v : y_train) counts[static_cast<int>(v)]++;

    // Posterior mean cutpoints (exp-scale = hazard increment)
    const int num_cuts = K - 1;
    std::vector<double> gamma_mean(num_cuts, 0.0);
    for (int s = 0; s < S; ++s)
        for (int k = 0; k < num_cuts; ++k)
            gamma_mean[k] += result.cloglog_cutpoint_samples[static_cast<size_t>(s) * num_cuts + k];
    for (int k = 0; k < num_cuts; ++k) gamma_mean[k] /= S;

    std::cout << "\n=== " << tag << ": Cloglog BART smoke test (K=" << K << ") ===\n"
              << "  n_train=" << n_train << "  n_test=" << n_test
              << "  p=" << p << "  trees=" << num_trees
              << "  GFR=" << num_gfr << "  MCMC=" << num_mcmc << "  samples=" << S << "\n";

    std::cout << "  Category counts in training: ";
    for (int k = 0; k < K; ++k) std::cout << k << ":" << counts[k] << " ";
    std::cout << "\n";

    std::cout << std::fixed << std::setprecision(4)
              << "  Posterior mean gamma (cutpoints): ";
    for (int k = 0; k < num_cuts; ++k) std::cout << gamma_mean[k] << " ";
    std::cout << "\n";

    std::cout << "  y_hat_train finite: "
              << (all_finite(result.y_hat_train) ? "yes" : "NO — BUG") << "\n"
              << "  cloglog_cutpoint_samples size: "
              << result.cloglog_cutpoint_samples.size()
              << "  (expected " << num_cuts * S << ")\n"
              << "  sigma2_global_samples empty: "
              << (result.sigma2_global_samples.empty() ? "yes" : "NO — BUG") << "\n"
              << "  ForestContainer samples: " << result.forest_container->NumSamples() << "\n";
}

// ── 5. Mean + variance-forest smoke test ─────────────────────────────────────

static void run_mean_varforest_smoke_test()
{
    constexpr int n_train = 500, n_test = 100, p = 5;
    constexpr int num_trees = 50, num_trees_var = 50;
    constexpr int num_gfr = 10, num_burnin = 50, num_mcmc = 100;

    std::vector<double> X_train, y_train, s_true_train;
    std::vector<double> X_test,  s_true_test;
    make_mean_varforest_data(n_train, p, X_train, y_train, s_true_train);
    // Generate test X independently for s_true_test
    {
        std::vector<double> y_test_unused;
        make_mean_varforest_data(n_test, p, X_test, y_test_unused, s_true_test, /*seed=*/56);
    }

    StochTree::BARTConfig config;
    config.num_trees              = num_trees;
    config.num_gfr                = num_gfr;
    config.num_burnin             = num_burnin;
    config.num_mcmc               = num_mcmc;
    config.keep_gfr               = false;
    config.keep_burnin            = false;
    config.include_variance_forest = true;
    config.num_trees_variance      = num_trees_var;
    config.sample_sigma2_global    = false;  // variance forest takes over
    config.sample_sigma2_leaf      = true;
    config.random_seed             = 42;

    StochTree::BARTData data;
    data.X_train = X_train.data(); data.n_train = n_train; data.p = p;
    data.y_train = y_train.data();
    data.X_test  = X_test.data();  data.n_test  = n_test;

    StochTree::BARTResult result;
    StochTree::BARTFit(&result, config, data);

    int S = result.num_total_samples;

    // Posterior mean of predicted std on test set
    std::vector<double> s_hat(n_test, 0.0);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < n_test; ++i)
            s_hat[i] += std::sqrt(result.sigma2_x_hat_test[static_cast<size_t>(s) * n_test + i]);
    for (int i = 0; i < n_test; ++i) s_hat[i] /= S;

    double rmse_s = 0.0, s_bar = 0.0;
    for (double v : s_true_test) s_bar += v;
    s_bar /= n_test;
    for (int i = 0; i < n_test; ++i) { double e = s_hat[i] - s_true_test[i]; rmse_s += e * e; }
    rmse_s = std::sqrt(rmse_s / n_test);
    double rmse_null = 0.0;
    for (int i = 0; i < n_test; ++i) { double e = s_bar - s_true_test[i]; rmse_null += e * e; }
    rmse_null = std::sqrt(rmse_null / n_test);

    std::cout << "\n=== mean+varforest: Mean + variance-forest smoke test ===\n"
              << "  n_train=" << n_train << "  n_test=" << n_test
              << "  p=" << p << "  trees=" << num_trees
              << "  var_trees=" << num_trees_var
              << "  GFR=" << num_gfr << "  burnin=" << num_burnin
              << "  MCMC=" << num_mcmc << "  samples=" << S << "\n";
    std::cout << std::fixed << std::setprecision(4)
              << "  y_hat_train finite: "
              << (all_finite(result.y_hat_train) ? "yes" : "NO — BUG") << "\n"
              << "  Test RMSE for E[sqrt(sigma2_x)]: " << rmse_s
              << "  (null: " << rmse_null << ")\n"
              << "  ForestContainer (mean) samples:     "
              << result.forest_container->NumSamples() << "\n"
              << "  ForestContainer (variance) samples: "
              << result.variance_forest_container->NumSamples() << "\n";
}

// ── 6 & 7. Leaf regression smoke test ────────────────────────────────────────

static void run_leaf_reg_smoke_test(int basis_dim = 1)
{
    const char* tag = (basis_dim == 1) ? "leaf-reg" : "leaf-reg-mv";
    constexpr int n_train = 400, n_test = 100, p = 5;
    constexpr int num_trees = 50, num_gfr = 10, num_burnin = 50, num_mcmc = 100;

    std::vector<double> X_train, y_train, X_test, basis_train, basis_test;
    make_leaf_reg_data(n_train, n_test, p, basis_dim,
                       X_train, y_train, X_test, basis_train, basis_test);

    StochTree::BARTConfig config;
    config.num_trees      = num_trees;
    config.num_gfr        = num_gfr;
    config.num_burnin     = num_burnin;
    config.num_mcmc       = num_mcmc;
    config.keep_gfr       = false;
    config.keep_burnin    = false;
    config.leaf_model     = (basis_dim > 1)
                            ? StochTree::LeafModel::MultivariateRegression
                            : StochTree::LeafModel::UnivariateRegression;
    // Leaf scale sampling: enabled for univariate, disabled for multivariate.
    config.sample_sigma2_leaf = (basis_dim == 1);
    config.random_seed    = 42;

    StochTree::BARTData data;
    data.X_train     = X_train.data();     data.n_train   = n_train; data.p = p;
    data.y_train     = y_train.data();
    data.X_test      = X_test.data();      data.n_test    = n_test;
    data.basis_train = basis_train.data(); data.basis_dim = basis_dim;
    data.basis_test  = basis_test.data();

    StochTree::BARTResult result;
    StochTree::BARTFit(&result, config, data);

    int S = result.num_total_samples;

    std::cout << "\n=== " << tag << ": Leaf regression smoke test (basis_dim=" << basis_dim << ") ===\n"
              << "  n_train=" << n_train << "  n_test=" << n_test
              << "  p=" << p << "  basis_dim=" << basis_dim
              << "  trees=" << num_trees
              << "  GFR=" << num_gfr << "  burnin=" << num_burnin
              << "  MCMC=" << num_mcmc << "  samples=" << S << "\n";
    std::cout << std::fixed << std::setprecision(4)
              << "  Train RMSE: " << rmse(result.y_hat_train, y_train, n_train, S) << "\n"
              << "  y_hat_train finite: "
              << (all_finite(result.y_hat_train) ? "yes" : "NO — BUG") << "\n"
              << "  y_hat_test  finite: "
              << (all_finite(result.y_hat_test)  ? "yes" : "NO — BUG") << "\n"
              << "  leaf_scale samples: " << result.leaf_scale_samples.size() << "\n"
              << "  ForestContainer samples: " << result.forest_container->NumSamples() << "\n";
}

// ── 8. Random effects (intercept-only) smoke test ────────────────────────────

static void run_rfx_smoke_test()
{
    constexpr int n_train = 400, n_test = 80, p = 5, num_groups = 10;
    constexpr int num_trees = 50, num_gfr = 10, num_burnin = 50, num_mcmc = 100;

    std::vector<double> X_train, y_train, X_test, y_test_unused;
    std::vector<int> groups_train, groups_test;
    make_rfx_data(n_train, p, num_groups, X_train, y_train, groups_train);
    make_rfx_data(n_test,  p, num_groups, X_test,  y_test_unused, groups_test, /*seed=*/77);

    StochTree::BARTConfig config;
    config.num_trees            = num_trees;
    config.num_gfr              = num_gfr;
    config.num_burnin           = num_burnin;
    config.num_mcmc             = num_mcmc;
    config.keep_gfr             = false;
    config.keep_burnin          = false;
    config.rfx_model_spec       = StochTree::RFXModelSpec::InterceptOnly;
    config.rfx_num_components   = 1;
    config.sample_sigma2_global = true;
    config.sample_sigma2_leaf   = true;
    config.random_seed          = 42;

    StochTree::BARTData data;
    data.X_train    = X_train.data();    data.n_train    = n_train; data.p = p;
    data.y_train    = y_train.data();
    data.X_test     = X_test.data();     data.n_test     = n_test;
    data.rfx_groups = groups_train.data();
    data.rfx_groups_test = groups_test.data();

    StochTree::BARTResult result;
    StochTree::BARTFit(&result, config, data);

    int S = result.num_total_samples;

    // Posterior mean of xi (group parameters) — layout: sample × group × component
    std::vector<double> xi_mean(num_groups, 0.0);
    if (result.rfx_container) {
        const std::vector<double>& xi = result.rfx_container->GetXi();
        for (int s = 0; s < S; ++s)
            for (int g = 0; g < num_groups; ++g)
                xi_mean[g] += xi[static_cast<size_t>(s) * num_groups + g];
        for (int g = 0; g < num_groups; ++g) xi_mean[g] /= S;
    }

    std::cout << "\n=== rfx: Random-effects (intercept-only) smoke test ===\n"
              << "  n_train=" << n_train << "  n_test=" << n_test
              << "  p=" << p << "  groups=" << num_groups
              << "  trees=" << num_trees
              << "  GFR=" << num_gfr << "  burnin=" << num_burnin
              << "  MCMC=" << num_mcmc << "  samples=" << S << "\n";
    std::cout << std::fixed << std::setprecision(4)
              << "  Train RMSE: " << rmse(result.y_hat_train, y_train, n_train, S) << "\n"
              << "  y_hat_train finite: "
              << (all_finite(result.y_hat_train) ? "yes" : "NO — BUG") << "\n"
              << "  rfx_container non-null: "
              << (result.rfx_container ? "yes" : "NO — BUG") << "\n"
              << "  rfx_container samples: "
              << (result.rfx_container ? result.rfx_container->NumSamples() : -1) << "\n"
              << "  rfx_container groups: "
              << (result.rfx_container ? result.rfx_container->NumGroups() : -1) << "\n"
              << "  Posterior mean xi (groups 0-4): ";
    for (int g = 0; g < std::min(5, num_groups); ++g)
        std::cout << xi_mean[g] << " ";
    std::cout << "\n"
              << "  ForestContainer samples: " << result.forest_container->NumSamples() << "\n";
}

// ── Wall-time table ──────────────────────────────────────────────────────────

struct Scenario {
    const char* label;
    int n, p, num_trees, num_gfr, num_mcmc, num_chains;
    StochTree::LinkFunction link      = StochTree::LinkFunction::Identity;
    bool variance_forest              = false;
    int  num_trees_variance           = 0;
    int  cloglog_K                    = 2;   ///< categories (link==Cloglog only)
    bool rfx                          = false;
    int  rfx_num_groups               = 0;   ///< (rfx==true only)
    bool leaf_reg                     = false;
};

static double time_bartfit(const Scenario& s, int reps = 3) {
    const bool is_cloglog = (s.link == StochTree::LinkFunction::Cloglog);
    const bool is_probit  = (s.link == StochTree::LinkFunction::Probit);

    std::vector<double> X, y, X_dummy, s_dummy, basis;
    std::vector<int> groups;

    if (is_cloglog) {
        make_ordinal_data(s.n, s.p, s.cloglog_K, X, y);
    } else if (s.rfx) {
        make_rfx_data(s.n, s.p, s.rfx_num_groups, X, y, groups);
    } else if (s.leaf_reg) {
        std::vector<double> basis_test_dummy;
        make_leaf_reg_data(s.n, 0, s.p, 1, X, y, X_dummy, basis, basis_test_dummy);
    } else if (is_probit) {
        make_binary_data(s.n, s.p, X, y);
    } else if (s.variance_forest) {
        make_heterosked_data(s.n, s.p, X, y, s_dummy);
    } else {
        make_continuous_data(s.n, 0, s.p, X, y, X_dummy);
    }

    StochTree::BARTConfig cfg;
    cfg.num_trees              = s.num_trees;
    cfg.num_gfr                = s.num_gfr;
    cfg.num_burnin             = 0;
    cfg.num_mcmc               = s.num_mcmc;
    cfg.num_chains             = s.num_chains;
    cfg.link_function          = s.link;
    cfg.include_variance_forest = s.variance_forest;
    cfg.num_trees_variance      = s.num_trees_variance;
    cfg.sample_sigma2_global   = (!s.variance_forest && !is_probit && !is_cloglog);
    cfg.cloglog_num_categories = s.cloglog_K;
    cfg.random_seed            = 99;
    if (s.rfx)
        cfg.rfx_model_spec = StochTree::RFXModelSpec::InterceptOnly;
    if (s.leaf_reg)
        cfg.leaf_model = StochTree::LeafModel::UnivariateRegression;

    StochTree::BARTData data;
    data.X_train = X.data(); data.n_train = s.n; data.p = s.p; data.y_train = y.data();
    if (s.rfx) {
        data.rfx_groups = groups.data();
    }
    if (s.leaf_reg) {
        data.basis_train = basis.data();
        data.basis_dim   = 1;
    }

    double total_ms = 0.0;
    for (int r = 0; r < reps; ++r) {
        StochTree::BARTResult result;
        auto t0 = Clock::now();
        StochTree::BARTFit(&result, cfg, data);
        total_ms += elapsed_ms(t0);
    }
    return total_ms / reps;
}

static void run_timing_table()
{
    using L = StochTree::LinkFunction;
    std::vector<Scenario> scenarios = {
        // label                      n     p    T   gfr  mcmc  ch  link          var    Tv    K    rfx  rg   lr
        {"GFR-only  (id)",          500,  10,  200,  10,   0,   1, L::Identity, false,  0,    2, false,  0, false},
        {"MCMC-only (id)",          500,  10,  200,   0, 100,   1, L::Identity, false,  0,    2, false,  0, false},
        {"Warm-start (id)",         500,  10,  200,   5, 100,   1, L::Identity, false,  0,    2, false,  0, false},
        {"Multi-chain (id)",        500,  10,  200,   3,  50,   3, L::Identity, false,  0,    2, false,  0, false},
        {"Large n (id)",           2000,  10,  200,   5, 100,   1, L::Identity, false,  0,    2, false,  0, false},
        {"GFR-only  (probit)",      500,  10,  200,  10,   0,   1, L::Probit,   false,  0,    2, false,  0, false},
        {"Warm-start (probit)",     500,  10,  200,   5, 100,   1, L::Probit,   false,  0,    2, false,  0, false},
        {"Warm-start (cloglog-2)",  500,  10,  200,   5, 100,   1, L::Cloglog,  false,  0,    2, false,  0, false},
        {"Warm-start (cloglog-3)",  500,  10,  200,   5, 100,   1, L::Cloglog,  false,  0,    3, false,  0, false},
        {"Warm-start (het)",        500,  10,    0,   5, 100,   1, L::Identity, true,  50,    2, false,  0, false},
        {"Warm-start (mean+het)",   500,  10,  200,   5, 100,   1, L::Identity, true,  50,    2, false,  0, false},
        {"Warm-start (leaf-reg)",   500,  10,  200,   5, 100,   1, L::Identity, false,  0,    2, false,  0, true },
        {"Warm-start (rfx-10g)",    500,  10,  200,   5, 100,   1, L::Identity, false,  0,    2, true,  10, false},
    };

    constexpr int reps = 3;

    std::cout << "\n=== Wall-time table (avg of " << reps << " reps) ===\n";
    std::cout << std::left  << std::setw(30) << "Scenario"
              << std::right << std::setw(12) << "BARTFit(ms)" << "\n"
              << std::string(44, '-') << "\n";

    for (const auto& s : scenarios) {
        double ms = time_bartfit(s, reps);
        std::string label = std::string(s.label)
            + " n=" + std::to_string(s.n)
            + " T=" + std::to_string(s.num_trees)
            + " S=" + std::to_string(s.num_gfr + s.num_mcmc * s.num_chains);
        std::cout << std::left  << std::setw(30) << label
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(12) << ms << "\n";
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────
//
// Usage:
//   ./build/debug_bart                  — all smoke tests
//   ./build/debug_bart --model <name>   — single model (see list above)
//   ./build/debug_bart --timing         — all smoke tests + wall-time table
//   ./build/debug_bart --model rfx --timing  — single test + timing table

int main(int argc, char* argv[])
{
    std::string model = "all";
    bool run_timing   = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--timing") {
            run_timing = true;
        } else if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout <<
                "Usage: debug_bart [--model <name>] [--timing]\n"
                "Models: identity probit varforest cloglog ordinal\n"
                "        mean+varforest leaf-reg leaf-reg-mv rfx all\n";
            return 0;
        }
    }

    auto run = [&](const std::string& name) {
        return model == "all" || model == name;
    };

    if (run("identity"))        run_identity_smoke_test();
    if (run("probit"))          run_probit_smoke_test();
    if (run("varforest"))       run_varforest_smoke_test();
    if (run("cloglog"))         run_cloglog_smoke_test(2);
    if (run("ordinal"))         run_cloglog_smoke_test(3);
    if (run("mean+varforest"))  run_mean_varforest_smoke_test();
    if (run("leaf-reg"))        run_leaf_reg_smoke_test(1);
    if (run("leaf-reg-mv"))     run_leaf_reg_smoke_test(2);
    if (run("rfx"))             run_rfx_smoke_test();

    if (run_timing) run_timing_table();

    std::cout << "\nDone.\n";
    return 0;
}
