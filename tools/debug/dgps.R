dgp_levels <- c("dgp_prediction_partitioned_lm", "dgp_prediction_step_function")

dgp_prediction_partitioned_lm <- function(n, p_x, p_w, snr = NULL) {
    X <- matrix(runif(n*p_x), ncol = p_x)
    W <- matrix(runif(n*p_w), ncol = p_w)
    f_XW <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*W[,1]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*W[,1]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*W[,1]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*W[,1])
    )
    if (!is.null(snr)) {
        if (snr > 0) {
            noise_sd <- sd(f_XW) / snr
            snr_used <- snr
        } else {
            noise_sd <- 1
            snr_used <- sd(f_XW) / noise_sd
        }
    } else {
        noise_sd <- 1
        snr_used <- sd(f_XW) / noise_sd
    }
    y <- f_XW + rnorm(n, 0, noise_sd)
    return(list(has_basis=T,X=X,W=W,y=y,noise_sd=noise_sd,snr=snr_used))
}

dgp_prediction_step_function <- function(n, p_x, snr = NULL) {
    X <- matrix(runif(n*p_x), ncol = p_x)
    f_XW <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
    )
    if (!is.null(snr)) {
        if (snr > 0) {
            noise_sd <- sd(f_XW) / snr
            snr_used <- snr
        } else {
            noise_sd <- 1
            snr_used <- sd(f_XW) / noise_sd
        }
    } else {
        noise_sd <- 1
        snr_used <- sd(f_XW) / noise_sd
    }
    y <- f_XW + rnorm(n, 0, noise_sd)
    return(list(has_basis=F,X=X,W=NULL,y=y,noise_sd=noise_sd,snr=snr_used))
}
