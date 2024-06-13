library(stochtree)

# Generate data with a continuous treatment
n <- 500
snr <- 3
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
x4 <- rnorm(n)
x5 <- rnorm(n)
X <- cbind(x1,x2,x3,x4,x5)
p <- ncol(X)
mu_x <- 1 + 2*x1 - 4*(x2 < 0) + 4*(x2 >= 0) + 3*(abs(x3) - sqrt(2/pi))
tau_x <- 1 + 2*x4
u <- runif(n)
pi_x <- ((mu_x-1)/4) + 4*(u-0.5)
Z <- pi_x + rnorm(n,0,1)
E_XZ <- mu_x + Z*tau_x
y <- E_XZ + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
X <- as.data.frame(X)

# Split data into test and train sets
test_set_pct <- 0.2
n_test <- round(test_set_pct*n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = F))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds,]
X_train <- X[train_inds,]
pi_test <- pi_x[test_inds]
pi_train <- pi_x[train_inds]
Z_test <- Z[test_inds]
Z_train <- Z[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]
mu_test <- mu_x[test_inds]
mu_train <- mu_x[train_inds]
tau_test <- tau_x[test_inds]
tau_train <- tau_x[train_inds]

# Run continuous treatment BCF
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 1000
num_samples <- num_gfr + num_burnin + num_mcmc
bcf_model_warmstart <- bcf(
    X_train = X_train, Z_train = Z_train, y_train = y_train, pi_train = pi_train, 
    X_test = X_test, Z_test = Z_test, pi_test = pi_test, 
    num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc, 
    sample_sigma_leaf_mu = F, sample_sigma_leaf_tau = F, verbose = T
)

# Inspect results
mu_hat_train <- rowMeans(bcf_model_warmstart$mu_hat_train)
tau_hat_train <- rowMeans(bcf_model_warmstart$tau_hat_train)
mu_hat_test <- rowMeans(bcf_model_warmstart$mu_hat_test)
tau_hat_test <- rowMeans(bcf_model_warmstart$tau_hat_test)
plot(mu_train, mu_hat_train); abline(0,1,lwd=3,lty=3,col="red")
plot(tau_train, tau_hat_train); abline(0,1,lwd=3,lty=3,col="red")
plot(mu_test, mu_hat_test); abline(0,1,lwd=3,lty=3,col="red")
plot(tau_test, tau_hat_test); abline(0,1,lwd=3,lty=3,col="red")
