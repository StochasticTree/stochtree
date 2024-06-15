# Create the stochtree_cran folder
cran_dir <- "stochtree_cran"
if (dir.exists(cran_dir)) {
    # cran_subfolder_files <- list.files(cran_dir, recursive = TRUE, full.names = TRUE)
    unlink(cran_dir, recursive = TRUE)
}