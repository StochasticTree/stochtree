library(stochtree)

df <- read.csv("debug/data/heterosked_train.csv")
y <- df[,"y"]
X <- df[,c('X1','X2','X3','X4','X5','X6','X7','X8','X9','X10')]

num_gfr <- 0
num_burnin <- 0
num_mcmc <- 10
general_params <- list(random_seed = 1234, standardize = F, sample_sigma2_global = T)
bart_model <- stochtree::bart(
    X_train = X, y_train = y, 
    num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc, 
    general_params = general_params
)

rowMeans(bart_model$y_hat_train)[1:20]
bart_model$sigma2_global_samples