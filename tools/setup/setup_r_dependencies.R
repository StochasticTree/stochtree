################################################################################
## This script is a modified version of the setup-r-dependencies Github action
## for local use and debugging. The source for the action is:
## https://github.com/r-lib/actions/blob/v2-branch/setup-r-dependencies/action.yaml
################################################################################

# Set site library path
cat("::group::Set site library path\n")
if (Sys.getenv("RENV_PROJECT") != "") {
    message("renv project detected, no need to set R_LIBS_SITE")
    cat(sprintf("R_LIB_FOR_PAK=%s\n", .libPaths()[1]), file = Sys.getenv("GITHUB_ENV"), append = TRUE)
    q("no")
}
lib <- .libPaths()[[1]]
if (lib == "") {
    lib <- file.path(dirname(.Library), "site-library")
    Sys.setenv(R_LIBS_SITE = strsplit(lib, .Platform$path.sep)[[1]][[1]])
    Sys.setenv(R_LIB_FOR_PAK = strsplit(lib, .Platform$path.sep)[[1]][[1]])
    message("Setting R_LIBS_SITE to ", lib)
    message("Setting R_LIB_FOR_PAK to ", lib)
} else {
    message("R_LIBS_SITE is already set to ", lib)
    Sys.setenv(R_LIB_FOR_PAK = strsplit(lib, .Platform$path.sep)[[1]][[1]])
    message("R_LIB_FOR_PAK is now set to ", lib)
}
cat("::endgroup::\n")

# Install pak
cat("::group::Install pak\n")
lib <- Sys.getenv("R_LIB_FOR_PAK")
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
install.packages("pak", lib = lib, repos = sprintf(
    "https://r-lib.github.io/p/pak/%s/%s/%s/%s",
    "stable",
    .Platform$pkgType,
    R.Version()$os,
    R.Version()$arch
))
cat("::endgroup::\n")

# Dependency resolution
cat("::group::Dependency resolution\n")
cat("os-version=", sessionInfo()$running, "\n", sep = "", append = TRUE)
r_version <-
    if (grepl("development", R.version.string)) {
        pdf(tempfile())
        ge_ver <- attr(recordPlot(), "engineVersion")
        dev.off()
        paste0("R version ", getRversion(), " (ge:", ge_ver, "; iid:", .Internal(internalsID()), ")")
    } else {
        R.version.string
    }
cat("r-version=", r_version, "\n", sep = "", append = TRUE)
needs <- sprintf("Config/Needs/%s", strsplit("", "[[:space:],]+")[[1]])
deps <- strsplit("any::cpp11, any::R6, any::knitr, any::rmarkdown, any::Matrix, any::tgp, any::MASS, any::mvtnorm, any::ggplot2, any::latex2exp, any::testthat, any::sessioninfo", "[[:space:],]+")[[1]]
extra_deps <- strsplit("any::testthat, any::decor, github::StochasticTree/stochtree-r", "[[:space:],]+")[[1]]
dir.create("install_temp", showWarnings=FALSE)
Sys.setenv("PKGCACHE_HTTP_VERSION" = "2")
library(pak, lib.loc = Sys.getenv("R_LIB_FOR_PAK"))
pak::lockfile_create(
    c(deps, extra_deps),
    lockfile = "install_temp/pkg.lock",
    upgrade = FALSE,
    dependencies = c(needs, "all"),
    lib = NULL
)
cat("::endgroup::\n")
cat("::group::Show Lockfile\n")
writeLines(readLines("install_temp/pkg.lock"))
cat("::endgroup::\n")

# Install/Update packages
cat("::group::Install/update packages\n")
Sys.setenv("PKGCACHE_HTTP_VERSION" = "2")
library(pak, lib.loc = Sys.getenv("R_LIB_FOR_PAK"))
pak::lockfile_install("install_temp/pkg.lock")

# Clean up temporary pkg.lock install directory
unlink("install_temp", recursive = TRUE)
cat("::endgroup::\n")
