#include <stochtree/ordinal_sampler.h>
#include <stdexcept>

namespace StochTree {

double OrdinalSampler::SampleTruncatedExponential(std::mt19937& gen, double rate, double low, double high) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  double u = unif(gen);
  if ((low <= 0.0) && (high <= 0.0)) {
    return sample_exponential(u, rate);
  } else if ((low <= 0.0) && (high > 0.0)) {
    return sample_truncated_exponential_high(u, rate, high);
  } else if ((low > 0.0) && (high <= 0.0)) {
    return sample_truncated_exponential_low(u, rate, low);
  } else {
    return sample_truncated_exponential_low_high(u, rate, low, high);
  }
}

void OrdinalSampler::UpdateLatentVariables(ForestDataset& dataset, Eigen::VectorXd& outcome, std::mt19937& gen) {
  // Get auxiliary data vectors
  const std::vector<double>& gamma = dataset.GetAuxiliaryDataVector(2);  // gamma cutpoints
  const std::vector<double>& lambda_hat = dataset.GetAuxiliaryDataVector(1);  // forest predictions: lambda_hat_i = sum_t lambda_t(x_i)
  std::vector<double>& Z = dataset.GetAuxiliaryDataVector(0);  // latent variables: z_i ~ TExp(e^{gamma[y_i] + lambda_hat_i}; 0, 1)

  int K = gamma.size() + 1;  // Number of ordinal categories
  int N = dataset.NumObservations();

  // Update truncated exponentials (stored in latent auxiliary data slot 0)
  // z_i ~ TExp(rate = e^{gamma[y_i] + lambda_hat_i}; 0, 1)
  // where y_i is the ordinal outcome for observation i: make sure y_i converted to {0, 1, ..., K-1}
  // and lambda_hat_i is the total forest prediction for observation i
  // If y_i = K-1 (last category), then we set z_i = 1.0 deterministically just for bookkeeping, we don't need it
  // We only need to sample latent z_i for y_i < K-1 (as z_i is only used in the likelihood for y_i < K-1)
  for (int i = 0; i < N; i++) {
    int y = static_cast<int>(outcome(i));
    if (y == K - 1) {
      Z[i] = 1.0;
    } else {
      double rate = std::exp(gamma[y] + lambda_hat[i]);
      Z[i] = SampleTruncatedExponential(gen, rate, 0.0, 1.0);
    }
  }
}

void OrdinalSampler::UpdateGammaParams(ForestDataset& dataset, Eigen::VectorXd& outcome, 
                                       double alpha_gamma, double beta_gamma, 
                                       double gamma_0, std::mt19937& gen) {
  // Get auxiliary data vectors
  std::vector<double>& gamma = dataset.GetAuxiliaryDataVector(2);  // cutpoints gamma_k's
  const std::vector<double>& Z = dataset.GetAuxiliaryDataVector(0);  // latent variables z_i's
  const std::vector<double>& lambda_hat = dataset.GetAuxiliaryDataVector(1);  // forest predictions: lambda_hat_i = sum_t lambda_t(x_i)

  int K = gamma.size() + 1;  // Number of ordinal categories
  int N = dataset.NumObservations();

  // Compute sufficient statistics A[k] and B[k] for gamma[k] update
  std::vector<double> A(K - 1, 0.0);
  std::vector<double> B(K - 1, 0.0);

  for (int i = 0; i < N; i++) {
    int y = static_cast<int>(outcome(i));
    if (y < K - 1) {
      A[y] += 1.0;
      B[y] += Z[i] * std::exp(lambda_hat[i]);
    }
    for (int k = 0; k < y; k++) {
      B[k] += std::exp(lambda_hat[i]);
    }
  }

  // Update gamma parameters using log-gamma sampling
  // First sample all gamma parameters
  for (int k = 0; k < static_cast<int>(gamma.size()); k++) {
    double shape = A[k] + alpha_gamma;
    double rate = B[k] + beta_gamma;
    double gamma_sample = gamma_sampler_.Sample(shape, rate, gen);
    gamma[k] = std::log(gamma_sample);
  }

  // Set the first gamma parameter to gamma_0 (e.g., 0) for identifiability
  // if (K > 2) {
    gamma[0] = gamma_0;
  // }
}

void OrdinalSampler::UpdateCumulativeExpSums(ForestDataset& dataset) {
  // Get auxiliary data vectors
  const std::vector<double>& gamma = dataset.GetAuxiliaryDataVector(2);  // cutpoints gamma_k's
  std::vector<double>& seg = dataset.GetAuxiliaryDataVector(3);    // seg_k = sum_{j=0}^{k-1} exp(gamma_j)

  // Update seg (sum of exponentials of gamma cutpoints)
  for (int j = 0; j < static_cast<int>(seg.size()); j++) {
    if (j == 0) {
      seg[j] = 0.0; // checked and it is correct
    } else {
      seg[j] = seg[j - 1] + std::exp(gamma[j - 1]);  // checked and it is correct
    }
  }
}

} // namespace StochTree
