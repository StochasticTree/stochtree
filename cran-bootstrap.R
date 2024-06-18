# Derived from a combination of
# 
# 1) The bootstrap.R files in the arrow-adbc suite of R packages,
# https://github.com/apache/arrow-adbc/tree/main/r, all of which are Apache
# licensed with the following copyright:
# Copyright 2022 The Apache Software Foundation
# 
# 2) LightGBM's build-cran-package.sh script, 
# https://github.com/microsoft/LightGBM/blob/master/build-cran-package.sh, 
# which is MIT licensed with the following copyright:
# Copyright (c) Microsoft Corporation

# Use xfun for find and replace in files / directories
# require(xfun)
# if (!require(xfun)) {
#     install.packages("xfun")
# }

# Create the stochtree_cran folder
cran_dir <- "stochtree_cran"
if (!dir.exists(cran_dir)) {
    dir.create(cran_dir, recursive = TRUE)
}

# Copy the "core" package files to CRAN folder
src_files <- list.files("src", recursive = TRUE, full.names = TRUE)
pybind_src_files <- list.files("src", pattern = "^(py_)", recursive = TRUE, full.names = TRUE)
r_src_files <- src_files[!(src_files %in% pybind_src_files)]
pkg_core_files <- c(
    ".Rbuildignore",
    "DESCRIPTION",
    "LICENSE",
    "LICENSE.md",
    list.files("man", recursive = TRUE, full.names = TRUE),
    "NAMESPACE",
    list.files("R", recursive = TRUE, full.names = TRUE),
    r_src_files, 
    list.files("vignettes", recursive = TRUE, full.names = TRUE)
)
pkg_core_files_dst <- file.path(cran_dir, pkg_core_files)
# Handle tests separately (move from test/R/ folder to tests/ folder)
test_files_src <- list.files("test/R", recursive = TRUE, full.names = TRUE)
test_files_dst <- file.path(cran_dir, gsub("test/R", "tests", test_files_src))
pkg_core_files <- c(pkg_core_files, test_files_src)
pkg_core_files_dst <- c(pkg_core_files_dst, test_files_dst)
if (all(file.exists(pkg_core_files))) {
    n_removed <- suppressWarnings(sum(file.remove(pkg_core_files_dst)))
    if (n_removed > 0) {
        cat(sprintf("Removed %d previously vendored files from temporary CRAN directory (%s)\n", n_removed, cran_dir))
    }
    
    cat(
        sprintf(
            "Copying core package files to CRAN subdirectory\n"
        )
    )
    
    # Recreate the directory structure
    dst_dirs <- unique(dirname(pkg_core_files_dst))
    for (dst_dir in dst_dirs) {
        if (!dir.exists(dst_dir)) {
            dir.create(dst_dir, recursive = TRUE)
        }
    }
    
    if (all(file.copy(pkg_core_files, pkg_core_files_dst))) {
        cat("All core package files successfully copied to CRAN subdirectory\n")
    } else {
        stop("Failed to copy all core package files")
    }
}

# Overwrite PKG_CPPFLAGS in src/Makevars
cran_makevars <- file.path(cran_dir, "src/Makevars")
makevars_lines <- readLines(cran_makevars)
makevars_lines[grep("^(PKG_CPPFLAGS)", makevars_lines)] <- "PKG_CPPFLAGS= -I$(PKGROOT)/src/include $(STOCHTREE_CPPFLAGS)"
writeLines(makevars_lines, cran_makevars)

# Copy fast_double_parser header to an include/ subdirectory of src/
header_folders <- c("nlohmann", "stochtree")
header_files_to_vendor_src <- c()
header_files_to_vendor_dst <- c()
for (header_folder in header_folders) {
    # Existing source files
    header_subfolder_src <- paste0("include/", header_folder)
    header_filenames_src <- list.files(header_subfolder_src, recursive = TRUE)
    header_src <- file.path(header_subfolder_src, header_filenames_src)
    header_files_to_vendor_src <- c(header_files_to_vendor_src, header_src)
    
    # Destination files
    header_subfolder_dst <- paste0(cran_dir, "/src/include/", header_folder)
    header_dst <- file.path(header_subfolder_dst, basename(header_src))
    header_files_to_vendor_dst <- c(header_files_to_vendor_dst, header_dst)
}

fast_double_parser_src <- file.path("deps/fast_double_parser/include/fast_double_parser.h")
fast_double_parser_dst <- file.path(cran_dir, "src/include/fast_double_parser.h")
if (file.exists(fast_double_parser_src)) {
    file_removed <- suppressWarnings(sum(file.remove(fast_double_parser_dst)) > 0)
    if (file_removed) {
        cat(sprintf("Removed previously vendored fast_double_parser.h file from src/include\n"))
    }
    
    cat(
        sprintf(
            "Vendoring fast_double_parser.h file from deps/fast_double_parser/include to src/include\n"
        )
    )
    
    # Recreate the directory structure
    dst_dir <- dirname(fast_double_parser_dst)
    if (!dir.exists(dst_dir)) {
        dir.create(dst_dir, recursive = TRUE)
    }

    if (file.copy(fast_double_parser_src, fast_double_parser_dst)) {
        cat("fast_double_parser.h header file successfully copied to src/include\n")
    } else {
        stop("Failed to vendor fast_double_parser.h header file")
    }
}

# Copy include/ headers to an include/ subdirectory of src/
header_folders <- c("nlohmann", "stochtree")
header_files_to_vendor_src <- c()
header_files_to_vendor_dst <- c()
for (header_folder in header_folders) {
    # Existing source files
    header_subfolder_src <- paste0("include/", header_folder)
    header_filenames_src <- list.files(header_subfolder_src, recursive = TRUE)
    header_src <- file.path(header_subfolder_src, header_filenames_src)
    header_files_to_vendor_src <- c(header_files_to_vendor_src, header_src)
    
    # Destination files
    header_subfolder_dst <- paste0(cran_dir, "/src/include/", header_folder)
    header_dst <- file.path(header_subfolder_dst, basename(header_src))
    header_files_to_vendor_dst <- c(header_files_to_vendor_dst, header_dst)
}

if (all(file.exists(header_files_to_vendor_src))) {
    n_removed <- suppressWarnings(sum(file.remove(header_files_to_vendor_dst)))
    if (n_removed > 0) {
        cat(sprintf("Removed %d previously vendored files from src/include\n", n_removed))
    }
    
    cat(
        sprintf(
            "Vendoring files from include/ to src/include\n"
        )
    )
    
    # Recreate the directory structure
    dst_dirs <- unique(dirname(header_files_to_vendor_dst))
    for (dst_dir in dst_dirs) {
        if (!dir.exists(dst_dir)) {
            dir.create(dst_dir, recursive = TRUE)
        }
    }
    
    if (all(file.copy(header_files_to_vendor_src, header_files_to_vendor_dst))) {
        cat("All include/ header files successfully copied to src/include\n")
    } else {
        stop("Failed to vendor all include/ header files")
    }
}

# Copy fmt headers to an include/ subdirectory of src/
fmt_header_files_to_vendor_src <- c()
fmt_header_files_to_vendor_dst <- c()
# Existing source files
fmt_header_subfolder_src <- "deps/fmt/include/fmt"
fmt_header_filenames_src <- list.files(fmt_header_subfolder_src, pattern = "\\.(h)$", recursive = TRUE)
fmt_header_files_to_vendor_src <- file.path(fmt_header_subfolder_src, fmt_header_filenames_src)
# Destination files
fmt_header_subfolder_dst <- "src/include/fmt"
fmt_header_files_to_vendor_dst <- file.path(cran_dir, fmt_header_subfolder_dst, basename(fmt_header_filenames_src))

if (all(file.exists(fmt_header_files_to_vendor_src))) {
    n_removed <- suppressWarnings(sum(file.remove(fmt_header_files_to_vendor_dst)))
    if (n_removed > 0) {
        cat(sprintf("Removed %d previously vendored files from src/include/fmt\n", n_removed))
    }
    
    cat(
        sprintf(
            "Vendoring files from deps/fmt/include/ to src/include/fmt\n"
        )
    )
    
    # Recreate the directory structure
    dst_dirs <- unique(dirname(fmt_header_files_to_vendor_dst))
    for (dst_dir in dst_dirs) {
        if (!dir.exists(dst_dir)) {
            dir.create(dst_dir, recursive = TRUE)
        }
    }
    
    if (all(file.copy(fmt_header_files_to_vendor_src, fmt_header_files_to_vendor_dst))) {
        cat("All deps/fmt/include/ header files successfully copied to src/include/fmt\n")
    } else {
        stop("Failed to vendor all deps/fmt/include/ header files")
    }
}

# Copy Eigen module headers to an include/Eigen subdirectory of src/
eigen_modules <- c("Cholesky", "Core", "Dense", "Eigenvalues", "Geometry", "Householder", "IterativeLinearSolvers", "Jacobi", "LU", "OrderingMethods", "QR", "SVD", "Sparse", "SparseCholesky", "SparseCore", "SparseLU", "SparseQR", "misc", "plugins")
eigen_files_to_vendor_src <- c()
eigen_files_to_vendor_dst <- c()
for (eigen_mod in eigen_modules) {
    # Existing source files
    eigen_module_subfolder_src <- paste0("deps/eigen/Eigen/src/", eigen_mod)
    # eigen_module_filenames_src <- list.files(eigen_module_subfolder_src, recursive = TRUE)
    # eigen_module_source_src <- file.path(eigen_module_subfolder_src, eigen_module_filenames_src)
    eigen_module_source_src <- list.files(eigen_module_subfolder_src, recursive = TRUE, full.names = TRUE)
    if (eigen_mod %in% c("misc", "plugins")) {
        eigen_files_to_vendor_src <- c(eigen_files_to_vendor_src, eigen_module_source_src)
    } else {
        eigen_module_header_src <- file.path("deps/eigen/Eigen", eigen_mod)
        eigen_files_to_vendor_src <- c(eigen_files_to_vendor_src, eigen_module_header_src, eigen_module_source_src)
    }
    
    # Destination files
    eigen_module_source_dst <- gsub("deps/eigen/Eigen", paste0(cran_dir, "/src/include/Eigen"), eigen_module_source_src)
    if (eigen_mod %in% c("misc", "plugins")) {
        eigen_files_to_vendor_dst <- c(eigen_files_to_vendor_dst, eigen_module_source_dst)
    } else {
        eigen_module_header_dst <- file.path(cran_dir, "src/include/Eigen", eigen_mod)
        eigen_files_to_vendor_dst <- c(eigen_files_to_vendor_dst, eigen_module_header_dst, eigen_module_source_dst)
    }
}

if (all(file.exists(eigen_files_to_vendor_src))) {
    n_removed <- suppressWarnings(sum(file.remove(eigen_files_to_vendor_dst)))
    if (n_removed > 0) {
        cat(sprintf("Removed %d previously vendored files from src/include/Eigen\n", n_removed))
    }
    
    cat(
        sprintf(
            "Vendoring files from deps/eigen to src/include/Eigen\n"
        )
    )
    
    # Recreate the directory structure
    dst_dirs <- unique(dirname(eigen_files_to_vendor_dst))
    for (dst_dir in dst_dirs) {
        if (!dir.exists(dst_dir)) {
            dir.create(dst_dir, recursive = TRUE)
        }
    }
    
    if (all(file.copy(eigen_files_to_vendor_src, eigen_files_to_vendor_dst))) {
        cat("All Eigen files successfully copied to src/include/Eigen\n")
    } else {
        stop("Failed to vendor all Eigen files")
    }
}
