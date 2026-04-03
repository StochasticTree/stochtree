/*!
 * Standalone BART driver for smoke-testing and profiling.
 *
 * Runs three sections in sequence:
 *   1. Identity-link BART smoke test (continuous outcome)
 *   2. Probit BART smoke test (binary outcome)
 *   3. Wall-time table across a set of scaling scenarios
 *
 * Use sections 1–2 to verify correctness after code changes.
 * Use section 3 as a profiling entry point — attach Instruments or perf
 * before launching, or rebuild with profiling flags and run directly.
 *
 * Build (debugging — unoptimized, full symbols):
 *   cmake -DBUILD_DEBUG_TARGETS=ON -DCMAKE_BUILD_TYPE=Debug \
 *         -B build && cmake --build build --target debug_bart
 *   ./build/debug_bart
 *
 * Build (profiling — optimized with symbols):
 *   cmake -DBUILD_DEBUG_TARGETS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
 *         -B build && cmake --build build --target debug_bart
 *   ./build/debug_bart --timing
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
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double elapsed_ms(Clock::time_point t0) {
    return Ms(Clock::now() - t0).count();
}

// ── Synthetic data ──────────────────────────────────────────────────────────

// Phi(x) = 0.5 * erfc(-x/sqrt(2))
static double Phi(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

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

// ── Summary statistics ──────────────────────────────────────────────────────

static double rmse(const std::vector<double>& pred,
                   const std::vector<double>& truth,
                   int n, int S)
{
    std::vector<double> y_bar(n, 0.0);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < n; ++i)
            y_bar[i] += pred[s * n + i];
    for (int i = 0; i < n; ++i) y_bar[i] /= S;

    double sse = 0.0;
    for (int i = 0; i < n; ++i) { double e = y_bar[i] - truth[i]; sse += e * e; }
    return std::sqrt(sse / n);
}

// ── 1. Identity BART smoke test ─────────────────────────────────────────────

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
    std::cout << "\n=== 1. Identity BART smoke test ===\n"
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

// ── 2. Probit BART smoke test ───────────────────────────────────────────────

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
    config.sample_sigma2_global = false;  // sigma2 fixed at 1 for probit
    config.sample_sigma2_leaf   = true;
    config.random_seed          = 42;

    StochTree::BARTData data;
    data.X_train = X_train.data(); data.n_train = n_train; data.p = p;
    data.y_train = y_train.data();
    data.X_test  = X_test.data();  data.n_test  = n_test;

    StochTree::BARTResult result;
    StochTree::BARTFit(&result, config, data);

    int S = result.num_total_samples;

    // Mean predicted probability on training set
    int n_pos = 0;
    for (double v : y_train) if (v > 0.5) n_pos++;
    double y_bar_prob = static_cast<double>(n_pos) / n_train;

    std::vector<double> p_hat(n_train, 0.0);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < n_train; ++i)
            p_hat[i] += Phi(result.y_hat_train[s * n_train + i]);
    for (int i = 0; i < n_train; ++i) p_hat[i] /= S;
    double mean_p_hat = std::accumulate(p_hat.begin(), p_hat.end(), 0.0) / n_train;

    std::cout << "\n=== 2. Probit BART smoke test ===\n"
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

// ── 3. Heteroskedastic variance-forest smoke test ───────────────────────────

// DGP: zero mean, variance driven by a partition on x0 × scale of x1.
// True conditional std: s(x) = {0.5, 1.0, 2.0, 3.0} × x1 (uniform [0,1]).
static void make_heterosked_data(int n, int p,
                                  std::vector<double>& X,
                                  std::vector<double>& y,
                                  std::vector<double>& s_true,  // true std, for reporting
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
        double x1 = (p > 1) ? X[n + i] : 0.5;
        double s  = (x0 < 0.25) ? 0.5 : (x0 < 0.5) ? 1.0 : (x0 < 0.75) ? 2.0 : 3.0;
        s *= x1;
        s_true[i] = s;
        y[i] = s * normal(rng);
    }
}

static void run_heterosked_smoke_test()
{
    constexpr int n_train = 500, n_test = 100, p = 5;
    constexpr int num_trees_var = 50, num_gfr = 10, num_burnin = 50, num_mcmc = 100;

    std::vector<double> X_train, y_train, s_true_train;
    std::vector<double> X_test,  y_test,  s_true_test;
    make_heterosked_data(n_train, p, X_train, y_train, s_true_train);
    make_heterosked_data(n_test,  p, X_test,  y_test,  s_true_test, 790);

    StochTree::BARTConfig config;
    config.num_trees               = 0;       // variance-only model
    config.num_gfr                 = num_gfr;
    config.num_burnin              = num_burnin;
    config.num_mcmc                = num_mcmc;
    config.keep_gfr                = false;
    config.keep_burnin             = false;
    config.include_variance_forest  = true;
    config.num_trees_variance       = num_trees_var;
    config.sample_sigma2_global     = false;
    config.sample_sigma2_leaf       = false;
    config.random_seed              = 42;

    StochTree::BARTData data;
    data.X_train = X_train.data(); data.n_train = n_train; data.p = p;
    data.y_train = y_train.data();
    data.X_test  = X_test.data();  data.n_test  = n_test;

    StochTree::BARTResult result;
    StochTree::BARTFit(&result, config, data);

    int S = result.num_total_samples;

    // Posterior mean of predicted std (= sqrt(sigma2_x)) on test set
    std::vector<double> s_hat(n_test, 0.0);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < n_test; ++i)
            s_hat[i] += std::sqrt(result.sigma2_x_hat_test[s * n_test + i]);
    for (int i = 0; i < n_test; ++i) s_hat[i] /= S;

    double rmse_s = 0.0;
    for (int i = 0; i < n_test; ++i) {
        double e = s_hat[i] - s_true_test[i];
        rmse_s += e * e;
    }
    rmse_s = std::sqrt(rmse_s / n_test);

    // Null RMSE: predict mean true std for all observations
    double s_bar = 0.0;
    for (double v : s_true_test) s_bar += v;
    s_bar /= n_test;
    double rmse_null = 0.0;
    for (int i = 0; i < n_test; ++i) {
        double e = s_bar - s_true_test[i];
        rmse_null += e * e;
    }
    rmse_null = std::sqrt(rmse_null / n_test);

    std::cout << "\n=== 3. Heteroskedastic variance-forest smoke test ===\n"
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

// ── 4. Wall-time table ──────────────────────────────────────────────────────

struct Scenario {
    const char* label;
    int n, p, num_trees, num_gfr, num_mcmc, num_chains;
    StochTree::LinkFunction link = StochTree::LinkFunction::Identity;
    bool variance_forest = false;
    int  num_trees_variance = 0;
};

static double time_bartfit(const Scenario& s, int reps = 3) {
    std::vector<double> X, y, X_dummy, s_dummy;
    if (s.link == StochTree::LinkFunction::Probit) {
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
    cfg.sample_sigma2_global   = (!s.variance_forest && s.link != StochTree::LinkFunction::Probit);
    cfg.random_seed            = 99;

    StochTree::BARTData data;
    data.X_train = X.data(); data.n_train = s.n; data.p = s.p; data.y_train = y.data();

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
    std::vector<Scenario> scenarios = {
        // label                    n     p    T    gfr  mcmc chains  link                            var_forest  T_var
        {"GFR-only  (id)",        500,  10,  200,  10,   0,   1, StochTree::LinkFunction::Identity, false, 0},
        {"MCMC-only (id)",        500,  10,  200,   0, 100,   1, StochTree::LinkFunction::Identity, false, 0},
        {"Warm-start (id)",       500,  10,  200,   5, 100,   1, StochTree::LinkFunction::Identity, false, 0},
        {"Multi-chain (id)",      500,  10,  200,   3,  50,   3, StochTree::LinkFunction::Identity, false, 0},
        {"Large n (id)",         2000,  10,  200,   5, 100,   1, StochTree::LinkFunction::Identity, false, 0},
        {"GFR-only  (probit)",    500,  10,  200,  10,   0,   1, StochTree::LinkFunction::Probit,   false, 0},
        {"Warm-start (probit)",   500,  10,  200,   5, 100,   1, StochTree::LinkFunction::Probit,   false, 0},
        {"Warm-start (het)",      500,  10,    0,   5, 100,   1, StochTree::LinkFunction::Identity, true,  50},
        {"Warm-start (mean+het)", 500,  10,  200,   5, 100,   1, StochTree::LinkFunction::Identity, true,  50},
    };

    constexpr int reps = 3;

    std::cout << "\n=== 4. Wall-time table (avg of " << reps << " reps) ===\n";
    std::cout << std::left  << std::setw(26) << "Scenario"
              << std::right << std::setw(12) << "BARTFit(ms)" << "\n"
              << std::string(40, '-') << "\n";

    for (const auto& s : scenarios) {
        double ms = time_bartfit(s, reps);
        std::string label = std::string(s.label)
            + " n=" + std::to_string(s.n)
            + " T=" + std::to_string(s.num_trees)
            + " S=" + std::to_string(s.num_gfr + s.num_mcmc * s.num_chains);
        std::cout << std::left  << std::setw(26) << label
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(12) << ms << "\n";
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────
//
// Usage:
//   ./build/debug_bart            — smoke tests only (fast; good for debug sessions)
//   ./build/debug_bart --timing   — smoke tests + wall-time table

int main(int argc, char* argv[])
{
    bool run_timing = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--timing") run_timing = true;

    run_identity_smoke_test();
    run_probit_smoke_test();
    run_heterosked_smoke_test();
    if (run_timing) run_timing_table();

    std::cout << "\nDone.\n";
    return 0;
}
