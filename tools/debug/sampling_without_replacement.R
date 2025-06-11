library(stochtree)

# Run sampler
a <- c(4,3,2,5,1,9,7)
p <- c(0.7,0.2,0.05,0.02,0.01,0.01,0.01)
num_samples <- 5
sample_without_replacement(as.integer(a), p, num_samples)
