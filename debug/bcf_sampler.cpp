/*!
 * Standalone BCFSampler driver for smoke-testing.
 *
 * Exercises BCFSamplerFit() across the core BCF configurations supported
 * by the C++ dispatch API.
 *
 * Supported models:
 *   identity      — continuous outcome, binary treatment, no propensity
 *   propensity    — same DGP, propensity score appended to mu forest
 *   adaptive      — adaptive coding (b0/b1 sampled), binary Z
 *   intercept     — sample_intercept=true (tau_0 sampled), no adaptive coding
 *   varforest     — BCF + variance forest
 *   no-intercept  — sample_intercept=false, no adaptive coding
 *
 * Usage:
 *   ./build/debug_bcf_sampler                     — all smoke tests
 *   ./build/debug_bcf_sampler --model identity    — single model
 *   ./build/debug_bcf_sampler --profile           — phase-timing breakdown
 *
 * Build:
 *   cmake -DBUILD_DEBUG_TARGETS=ON -B build && cmake --build build --target debug_bcf_sampler
 */

#include <stochtree/bcf.h>
#include <stochtree/bcf_sampler.h>

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

// ── Helpers ──────────────────────────────────────────────────────────────────

static bool all_finite(const std::vector<double>& v) {
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
}

static double vec_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// Posterior mean of a per-observation prediction array (n_obs × S, column-major).
static std::vector<double> posterior_mean(const std::vector<double>& arr, int n, int S) {
    std::vector<double> mu(n, 0.0);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < n; ++i)
            mu[i] += arr[static_cast<size_t>(s) * n + i];
    for (int i = 0; i < n; ++i) mu[i] /= S;
    return mu;
}

static double rmse(const std::vector<double>& pred_arr, const std::vector<double>& truth, int n, int S) {
    auto mu = posterior_mean(pred_arr, n, S);
    double sse = 0.0;
    for (int i = 0; i < n; ++i) { double e = mu[i] - truth[i]; sse += e * e; }
    return std::sqrt(sse / n);
}

// ── Data generators ───────────────────────────────────────────────────────────

// Friedman-like BCF DGP:
//   mu(X)  = sin(pi * X0 * X1) + 2*(X2 - 0.5)^2 + X3 + 0.5*X4
//   tau(X) = 1 + X2 + X3
//   Z      ~ Bernoulli(Phi(pi_hat)) where pi_hat = Phi(0.5*X0 - 0.25*X1)
//   y      = mu(X) + tau(X)*Z + eps,  eps ~ N(0, 1)
static void make_bcf_data(int n_train, int n_test, int p,
                           std::vector<double>& X_train,
                           std::vector<double>& y_train,
                           std::vector<double>& Z_train,
                           std::vector<double>& pi_hat_train,
                           std::vector<double>& tau_true_train,
                           std::vector<double>& X_test,
                           std::vector<double>& Z_test,
                           std::vector<double>& pi_hat_test,
                           std::vector<double>& tau_true_test,
                           unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    auto erfc_Phi = [](double x) {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    };

    auto fill_obs = [&](int n, std::vector<double>& X,
                        std::vector<double>& y,
                        std::vector<double>& Z,
                        std::vector<double>& pi,
                        std::vector<double>& tau_true) {
        X.resize(static_cast<size_t>(n) * p);
        y.resize(n); Z.resize(n); pi.resize(n); tau_true.resize(n);

        for (int j = 0; j < p; ++j)
            for (int i = 0; i < n; ++i)
                X[j * n + i] = (j < 2) ? normal(rng) : unif(rng);

        for (int i = 0; i < n; ++i) {
            double x0 = X[i],
                   x1 = (p > 1) ? X[n + i]     : 0.0,
                   x2 = (p > 2) ? X[2 * n + i] : 0.5,
                   x3 = (p > 3) ? X[3 * n + i] : 0.5,
                   x4 = (p > 4) ? X[4 * n + i] : 0.5;

            double mu_i  = std::sin(M_PI * x0 * x1) + 2.0 * (x2 - 0.5) * (x2 - 0.5)
                         + x3 + 0.5 * x4;
            tau_true[i]  = 1.0 + x2 + x3;

            pi[i] = erfc_Phi(0.5 * x0 - 0.25 * x1);
            Z[i]  = (unif(rng) < pi[i]) ? 1.0 : 0.0;
            y[i]  = mu_i + tau_true[i] * Z[i] + normal(rng);
        }
    };

    std::vector<double> y_test_unused;
    fill_obs(n_train, X_train, y_train,        Z_train, pi_hat_train, tau_true_train);
    fill_obs(n_test,  X_test,  y_test_unused,  Z_test,  pi_hat_test,  tau_true_test);
}

// ── Shared BCFData / BCFConfig helpers ────────────────────────────────────────

struct BCFTestData {
    int n_train, n_test, p;
    std::vector<double> X_train, y_train, Z_train, pi_hat_train, tau_true_train;
    std::vector<double> X_test,  Z_test,  pi_hat_test,  tau_true_test;
};

static BCFTestData make_test_data(int n_train = 500, int n_test = 100, int p = 5,
                                   unsigned seed = 42)
{
    BCFTestData d;
    d.n_train = n_train; d.n_test = n_test; d.p = p;
    make_bcf_data(n_train, n_test, p,
                  d.X_train, d.y_train, d.Z_train, d.pi_hat_train, d.tau_true_train,
                  d.X_test,  d.Z_test,  d.pi_hat_test,  d.tau_true_test,
                  seed);
    return d;
}

static StochTree::BCFData fill_data(const BCFTestData& d,
                                     bool include_propensity = true,
                                     bool include_test = true)
{
    StochTree::BCFData data;
    data.X_train   = d.X_train.data(); data.n_train = d.n_train; data.p = d.p;
    data.y_train   = d.y_train.data();
    data.Z_train   = d.Z_train.data();
    if (include_propensity) data.pi_hat_train = d.pi_hat_train.data();
    if (include_test) {
        data.X_test      = d.X_test.data();  data.n_test = d.n_test;
        data.Z_test      = d.Z_test.data();
        if (include_propensity) data.pi_hat_test = d.pi_hat_test.data();
    }
    return data;
}

static StochTree::BCFConfig default_config(int num_gfr = 10, int num_burnin = 50,
                                            int num_mcmc = 100)
{
    StochTree::BCFConfig cfg;
    cfg.num_gfr    = num_gfr;
    cfg.num_burnin = num_burnin;
    cfg.num_mcmc   = num_mcmc;
    cfg.keep_gfr   = false;
    cfg.keep_burnin = false;
    cfg.mu_forest.num_trees  = 50;
    cfg.tau_forest.num_trees = 25;
    cfg.mu_forest.sample_sigma2_leaf  = true;
    cfg.tau_forest.sample_sigma2_leaf = false;
    cfg.sample_sigma2_global = true;
    cfg.sample_intercept     = true;
    cfg.adaptive_coding      = false;
    cfg.random_seed          = 42;
    return cfg;
}

// ── Print helpers ─────────────────────────────────────────────────────────────

static void print_header(const std::string& label,
                          int n_train, int n_test, int p,
                          int T_mu, int T_tau, int num_gfr, int num_burnin, int num_mcmc, int S)
{
    std::cout << "\n=== " << label << " ===\n"
              << "  n_train=" << n_train << "  n_test=" << n_test
              << "  p=" << p
              << "  T_mu=" << T_mu << "  T_tau=" << T_tau
              << "  GFR=" << num_gfr << "  burnin=" << num_burnin
              << "  MCMC=" << num_mcmc << "  samples=" << S << "\n";
}

static void print_common(const StochTree::BCFResult& r, const BCFTestData& d)
{
    int S = r.num_total_samples;
    // tau_0_samples layout: treatment_dim × num_total_samples (scalar treatment: S entries)
    double tau_0_mean = vec_mean(r.tau_0_samples);

    std::cout << std::fixed << std::setprecision(4)
              << "  y_hat_train RMSE:     " << rmse(r.y_hat_train, d.y_train, d.n_train, S) << "\n"
              << "  y_hat_train finite:   " << (all_finite(r.y_hat_train) ? "yes" : "NO — BUG") << "\n"
              << "  tau_hat_train finite: " << (all_finite(r.tau_hat_train) ? "yes" : "NO — BUG") << "\n"
              << "  tau_0 posterior mean: " << tau_0_mean << "\n"
              << "  sigma2_global pmean:  " << vec_mean(r.sigma2_global_samples) << "\n"
              << "  sigma2_global empty:  " << (r.sigma2_global_samples.empty() ? "yes (BUG)" : "no") << "\n"
              << "  mu_forest samples:    " << (r.mu_forest_container  ? r.mu_forest_container->NumSamples()  : -1) << "\n"
              << "  tau_forest samples:   " << (r.tau_forest_container ? r.tau_forest_container->NumSamples() : -1) << "\n";
    if (!r.y_hat_test.empty())
        std::cout << "  y_hat_test finite:    " << (all_finite(r.y_hat_test) ? "yes" : "NO — BUG") << "\n";
}

// ── 1. Identity BCF (no propensity) ──────────────────────────────────────────

static void run_identity_smoke_test()
{
    auto d   = make_test_data();
    auto cfg = default_config();
    cfg.propensity_covariate = "none";

    auto data = fill_data(d, /*include_propensity=*/false);

    StochTree::BCFResult result;
    StochTree::BCFSamplerFit(&result, cfg, data);

    int S = result.num_total_samples;
    print_header("identity: BCF, no propensity score",
                 d.n_train, d.n_test, d.p,
                 cfg.mu_forest.num_trees, cfg.tau_forest.num_trees,
                 cfg.num_gfr, cfg.num_burnin, cfg.num_mcmc, S);
    print_common(result, d);
    std::cout << "  leaf_scale_mu pmean:  " << vec_mean(result.leaf_scale_mu_samples) << "\n"
              << "  leaf_scale_tau empty: " << (result.leaf_scale_tau_samples.empty() ? "yes" : "no (BUG — tau sample_sigma2_leaf=false)") << "\n"
              << "  b0/b1 empty:          " << (result.b0_samples.empty() ? "yes" : "no (BUG)") << "\n";
}

// ── 2. BCF with propensity score ──────────────────────────────────────────────

static void run_propensity_smoke_test()
{
    auto d   = make_test_data();
    auto cfg = default_config();
    cfg.propensity_covariate = "mu";  // default: pi_hat in mu-forest only

    auto data = fill_data(d, /*include_propensity=*/true);

    StochTree::BCFResult result;
    StochTree::BCFSamplerFit(&result, cfg, data);

    int S = result.num_total_samples;
    print_header("propensity: BCF, pi_hat in mu-forest",
                 d.n_train, d.n_test, d.p,
                 cfg.mu_forest.num_trees, cfg.tau_forest.num_trees,
                 cfg.num_gfr, cfg.num_burnin, cfg.num_mcmc, S);
    print_common(result, d);
}

// ── 3. BCF with adaptive coding ───────────────────────────────────────────────

static void run_adaptive_coding_smoke_test()
{
    auto d   = make_test_data();
    auto cfg = default_config();
    cfg.propensity_covariate = "mu";
    cfg.adaptive_coding      = true;
    cfg.b0_init              = -0.5;
    cfg.b1_init              =  0.5;
    cfg.coding_prior_var     =  0.5;

    auto data = fill_data(d, /*include_propensity=*/true);

    StochTree::BCFResult result;
    StochTree::BCFSamplerFit(&result, cfg, data);

    int S = result.num_total_samples;
    print_header("adaptive: BCF, adaptive coding (b0/b1 sampled)",
                 d.n_train, d.n_test, d.p,
                 cfg.mu_forest.num_trees, cfg.tau_forest.num_trees,
                 cfg.num_gfr, cfg.num_burnin, cfg.num_mcmc, S);
    print_common(result, d);
    std::cout << "  b0 posterior mean:    " << vec_mean(result.b0_samples)
              << "  (init=" << cfg.b0_init << ")\n"
              << "  b1 posterior mean:    " << vec_mean(result.b1_samples)
              << "  (init=" << cfg.b1_init << ")\n"
              << "  b0 samples count:     " << result.b0_samples.size() << "\n"
              << "  b1 samples count:     " << result.b1_samples.size() << "\n";
}

// ── 4. BCF without intercept (sample_intercept=false) ─────────────────────────

static void run_no_intercept_smoke_test()
{
    auto d   = make_test_data();
    auto cfg = default_config();
    cfg.propensity_covariate = "mu";
    cfg.sample_intercept     = false;

    auto data = fill_data(d, /*include_propensity=*/true);

    StochTree::BCFResult result;
    StochTree::BCFSamplerFit(&result, cfg, data);

    int S = result.num_total_samples;
    print_header("no-intercept: BCF, sample_intercept=false",
                 d.n_train, d.n_test, d.p,
                 cfg.mu_forest.num_trees, cfg.tau_forest.num_trees,
                 cfg.num_gfr, cfg.num_burnin, cfg.num_mcmc, S);
    print_common(result, d);
    std::cout << "  tau_0 samples (expect all zero): min="
              << *std::min_element(result.tau_0_samples.begin(), result.tau_0_samples.end())
              << "  max="
              << *std::max_element(result.tau_0_samples.begin(), result.tau_0_samples.end()) << "\n";
}

// ── 5. BCF with variance forest ───────────────────────────────────────────────

static void run_varforest_smoke_test()
{
    auto d   = make_test_data();
    auto cfg = default_config();
    cfg.propensity_covariate       = "mu";
    cfg.include_variance_forest    = true;
    cfg.num_trees_variance         = 25;
    cfg.alpha_variance             = 0.95;
    cfg.beta_variance              = 2.0;
    cfg.min_samples_leaf_variance  = 5;

    auto data = fill_data(d, /*include_propensity=*/true);

    StochTree::BCFResult result;
    StochTree::BCFSamplerFit(&result, cfg, data);

    int S = result.num_total_samples;
    print_header("varforest: BCF + variance forest",
                 d.n_train, d.n_test, d.p,
                 cfg.mu_forest.num_trees, cfg.tau_forest.num_trees,
                 cfg.num_gfr, cfg.num_burnin, cfg.num_mcmc, S);
    print_common(result, d);
    std::cout << "  sigma2_x_hat_train finite: "
              << (all_finite(result.sigma2_x_hat_train) ? "yes" : "NO — BUG") << "\n"
              << "  variance_forest samples:   "
              << (result.variance_forest_container
                    ? result.variance_forest_container->NumSamples() : -1) << "\n";
}

// ── Phase-profiling utility ───────────────────────────────────────────────────

static void run_phase_profile(const std::string& label,
                               StochTree::BCFConfig cfg,
                               const StochTree::BCFData& data,
                               int n_gfr, int n_mcmc, int reps = 3)
{
    cfg.num_gfr    = n_gfr;
    cfg.num_burnin = 0;
    cfg.num_mcmc   = n_mcmc;
    cfg.num_chains = 1;
    cfg.num_threads = 1;
    cfg.keep_gfr    = false;
    cfg.keep_burnin = false;

    double acc_ctor = 0, acc_gfr = 0, acc_mcmc = 0;

    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        StochTree::BCFSampler sampler(cfg, data);
        acc_ctor += elapsed_ms(t0);

        t0 = Clock::now();
        sampler.run_gfr(n_gfr);
        acc_gfr += elapsed_ms(t0);

        StochTree::BCFResult result;
        t0 = Clock::now();
        sampler.run_mcmc(n_mcmc, &result);
        acc_mcmc += elapsed_ms(t0);
    }

    double t_ctor = acc_ctor / reps;
    double t_gfr  = acc_gfr  / reps;
    double t_mcmc = acc_mcmc / reps;
    double total  = t_ctor + t_gfr + t_mcmc;

    std::cout << "\n── " << label << " (avg over " << reps << " reps) ──\n"
              << std::fixed << std::setprecision(2)
              << "  BCFSampler ctor: " << std::setw(8) << t_ctor
              << " ms  (" << std::setprecision(1) << t_ctor / total * 100 << "%)\n"
              << "  run_gfr (" << n_gfr << " iters): " << std::setprecision(2)
              << std::setw(8) << t_gfr
              << " ms  (" << std::setprecision(1) << t_gfr / total * 100 << "%)\n"
              << "  run_mcmc (" << n_mcmc << " iters): " << std::setprecision(2)
              << std::setw(8) << t_mcmc
              << " ms  (" << std::setprecision(1) << t_mcmc / total * 100 << "%)\n"
              << "  TOTAL:            " << std::setprecision(2)
              << std::setw(8) << total << " ms\n";
}

static void run_default_profile()
{
    constexpr int n = 500, p = 5, n_gfr = 10, n_mcmc = 100;

    auto d = make_test_data(n, 100, p);
    auto data = fill_data(d, /*include_propensity=*/true);

    auto cfg_base = default_config();
    cfg_base.propensity_covariate = "mu";

    auto cfg_adaptive = cfg_base;
    cfg_adaptive.adaptive_coding = true;

    std::cout << "\n=== Phase profiling: BCF variants ===\n"
              << "  n=" << n << "  p=" << p
              << "  n_gfr=" << n_gfr << "  n_mcmc=" << n_mcmc << "\n";

    run_phase_profile("BCF (no adaptive, intercept)",       cfg_base,     data, n_gfr, n_mcmc);
    run_phase_profile("BCF (adaptive coding, intercept)",  cfg_adaptive, data, n_gfr, n_mcmc);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    std::string model = "all";
    bool profile_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--profile") {
            profile_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout <<
                "Usage: debug_bcf_sampler [--model <name>] [--profile]\n"
                "Models: identity propensity adaptive no-intercept varforest all\n"
                "--profile: run per-phase timing breakdown\n";
            return 0;
        }
    }

    if (profile_mode) {
        run_default_profile();
        std::cout << "\nDone.\n";
        return 0;
    }

    auto run = [&](const std::string& name) {
        return model == "all" || model == name;
    };

    if (run("identity"))     run_identity_smoke_test();
    if (run("propensity"))   run_propensity_smoke_test();
    if (run("adaptive"))     run_adaptive_coding_smoke_test();
    if (run("no-intercept")) run_no_intercept_smoke_test();
    if (run("varforest"))    run_varforest_smoke_test();

    std::cout << "\nDone.\n";
    return 0;
}
