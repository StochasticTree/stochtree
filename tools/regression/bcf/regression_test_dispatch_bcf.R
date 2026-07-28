# Test case parameters
dgps <- 1:4
ns <- c(1000, 10000)
ps <- c(5, 20)
threads <- c(-1, 1)
varying_param_grid <- expand.grid(dgps, ns, ps, threads)
test_case_grid <- cbind(
    5, varying_param_grid[,2], varying_param_grid[,3], 
    10, 100, varying_param_grid[,1], 2.0, 0.2, varying_param_grid[,4]
)

# Run script for every case
script_path <- "tools/regression/bcf/individual_regression_test_bcf.R"
for (i in 1:nrow(test_case_grid)) {
    n_iter <- test_case_grid[i,1]
    n <- test_case_grid[i,2]
    p <- test_case_grid[i,3]
    num_gfr <- test_case_grid[i,4]
    num_mcmc <- test_case_grid[i,5]
    dgp_num <- test_case_grid[i,6]
    snr <- test_case_grid[i,7]
    test_set_pct <- test_case_grid[i,8]
    num_threads <- test_case_grid[i,9]
    system2(
        "Rscript", 
        args = c(script_path, n_iter, n, p, num_gfr, num_mcmc, dgp_num, snr, test_set_pct, num_threads)
    )
}
