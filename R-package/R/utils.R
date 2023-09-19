#' Utility function from lightGBM's R package.
#' Converts R-style list of parameters to a string of
#' the form "param1=-1,param2=yes,..."
#' 
#' Renamed here to reflect the naming convention of this package, but 
#' the code is the same.
#'
#' @param params List of parameters as inputs to either the dataset
#' creation process or the model training process
#'
#' @return String containing parameters and their values
#'
#' @examples
#' param_list <- list(param1 = -1, param2 = "yes")
#' param_string <- stochtree.params2str(param_list)
stochtree.params2str <- function(params) {

    if (!identical(class(params), "list")) {
        stop("params must be a list")
    }

    names(params) <- gsub(".", "_", names(params), fixed = TRUE)
    param_names <- names(params)
    ret <- list()

    # Perform key value join
    for (i in seq_along(params)) {

        # If a parameter has multiple values, join those values together with commas.
        # trimws() is necessary because format() will pad to make strings the same width
        val <- paste0(
            trimws(
                format(
                    x = unname(params[[i]])
                    , scientific = FALSE
                )
            )
            , collapse = ","
        )
        if (nchar(val) <= 0L) next # Skip join

        # Join key value
        pair <- paste0(c(param_names[[i]], val), collapse = "=")
        ret <- c(ret, pair)

    }

    if (length(ret) == 0L) {
        return("")
    }

    return(paste0(ret, collapse = " "))

}
