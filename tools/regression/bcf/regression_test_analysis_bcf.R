reg_test_dir <- "tools/regression/bcf/stochtree_bcf_r_results"
reg_test_files <- list.files(reg_test_dir, pattern = ".csv", full.names = T)

reg_test_df <- data.frame()
for (file in reg_test_files) {
    temp_df <- read.csv(file)
    reg_test_df <- rbind(reg_test_df, temp_df)
}

summary_df <- aggregate(
    cbind(outcome_rmse, outcome_coverage, treatment_effect_rmse, treatment_effect_coverage, runtime) ~ n + p + num_gfr + num_mcmc + dgp_num + snr + test_set_pct + num_threads, 
    data = reg_test_df, FUN = median, drop = TRUE
)

summary_file_output <- file.path(reg_test_dir, "stochtree_bcf_r_summary.csv")
write.csv(summary_df, summary_file_output, row.names = F)
