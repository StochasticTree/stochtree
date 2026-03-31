library(stochtree)

a <- c(4, 3, 2, 5, 1, 9, 7)
p <- c(0.7, 0.2, 0.05, 0.02, 0.01, 0.01, 0.01)
num_mc <- 100000
num_samples <- 5
results_stochtree <- matrix(NA_integer_, nrow = num_mc, ncol = num_samples)
results_base_R <- matrix(NA_integer_, nrow = num_mc, ncol = num_samples)

for (i in 1:num_mc) {
  results_stochtree[i, ] <- sample_without_replacement(
    as.integer(a),
    p,
    num_samples
  )
  results_base_R[i, ] <- sample(a, num_samples, replace = F, prob = p)
}


count_elems_stochtree <- rep(NA_integer_, length(a))
count_elems_base_R <- rep(NA_integer_, length(a))
for (i in 1:length(a)) {
  count_elems_stochtree[i] <- sum(rowSums(results_stochtree == a[i]))
  count_elems_base_R[i] <- sum(rowSums(results_base_R == a[i]))
}
count_elems_stochtree
count_elems_base_R
