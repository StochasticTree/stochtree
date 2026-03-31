library(microbenchmark)
library(stochtree)

# Run microbenchmark
num_elements <- 10000
a <- 1:num_elements
p <- runif(num_elements)
p <- p / sum(p)
num_samples <- 500
(bench_results <- microbenchmark(
  sample(a, num_samples, replace = F, prob = p),
  sample_without_replacement(as.integer(a), p, num_samples)
))
