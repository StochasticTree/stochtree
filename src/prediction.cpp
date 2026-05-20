/*! Copyright (c) 2026 by stochtree authors */
#include <stochtree/bcf.h>
#include <stochtree/bcf_sampler.h>
#include <stochtree/data.h>
#include <stochtree/distributions.h>
#include <stochtree/leaf_model.h>
#include <stochtree/linear_regression.h>
#include <stochtree/meta.h>
#include <stochtree/prediction.h>
#include <stochtree/probit.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <Eigen/Dense>
#include "stochtree/log.h"

namespace StochTree {

void location_scale_adjust_predictions(std::vector<double>& predictions, double location, double scale) {
  for (double& pred : predictions) {
    pred = pred * scale + location;
  }
}

void probability_transform_probit_predictions(std::vector<double>& predictions) {
  for (double& pred : predictions) {
    pred = norm_cdf(pred);
  }
}

void probability_transform_binary_cloglog_predictions(std::vector<double>& predictions) {
  for (double& pred : predictions) {
    pred = std::exp(-std::exp(pred));
  }
}

void probability_transform_ordinal_cloglog_predictions(std::vector<double>& predictions, std::vector<double>& output, double* cutpoints, int num_obs, int num_classes, int num_samples) {
  // Sequential ordinal cloglog model: P(Y=k) = prod_{j<k} S_j * (1 - S_k)
  // where S_k = exp(-exp(cutpoint_k + eta)).  The last class absorbs the
  // remaining survival probability: P(Y=K-1) = prod_{j<K-1} S_j.
  for (int j = 0; j < num_samples; j++) {
    for (int i = 0; i < num_obs; i++) {
      double eta = predictions[j * num_obs + i];
      double cumulative_survival = 1.0;
      double cumulative_prob = 0.0;
      for (int k = 0; k < num_classes - 1; k++) {
        double cutpoint = cutpoints[j * (num_classes - 1) + k];
        double S_k = std::exp(-std::exp(cutpoint + eta));
        double prob_k = cumulative_survival * (1.0 - S_k);
        output[j * num_classes * num_obs + k * num_obs + i] = prob_k;
        cumulative_prob += prob_k;
        cumulative_survival *= S_k;
      }
      output[j * num_classes * num_obs + (num_classes - 1) * num_obs + i] = 1.0 - cumulative_prob;
    }
  }
}

void class_transform_binary_outcome_predictions(std::vector<double>& predictions) {
  for (double& pred : predictions) {
    pred = pred >= 0.5 ? 1.0 : 0.0;
  }
}

/*!
 * \brief Assumes that class probabilities are stored in a column-major format in `predictions`, with dimensions num_obs x num_classes x num_samples, and transforms these into class predictions by taking the class with the highest predicted probability for each observation and sample.
 * The output is stored in `output`, which should be a pre-allocated vector of size num_obs x num_samples, also in column-major format (i.e., all predictions for the first sample are stored contiguously, followed by all predictions for the second sample, etc.).
 *
 * \param predictions Vector of class probabilities in column-major format with dimensions num_obs x num_classes x num_samples
 * \param output Pre-allocated vector to store class predictions in column-major format with dimensions num_obs x num_samples
 * \param num_obs Number of observations
 * \param num_classes Number of classes
 * \param num_samples Number of samples
 */
void class_transform_multiclass_outcome_predictions(std::vector<double>& predictions, std::vector<double>& output, int num_obs, int num_classes, int num_samples) {
  for (int j = 0; j < num_samples; j++) {
    for (int i = 0; i < num_obs; i++) {
      int predicted_class = 0;
      double max_prob = predictions[j * num_classes * num_obs + 0 * num_obs + i];
      for (int k = 1; k < num_classes; k++) {
        double prob_k = predictions[j * num_classes * num_obs + k * num_obs + i];
        if (prob_k > max_prob) {
          max_prob = prob_k;
          predicted_class = k;
        }
      }
      output[j * num_obs + i] = static_cast<double>(predicted_class);
    }
  }
}

/*!
 * \brief Internal helper function to average over the columns of a column-major 3d array. Works similarly to `np.mean(..., axis=1)` in numpy.
 *
 * \param input Vector representation of a 2d array stored in column-major order
 * \param output Empty vector allocated to store the 1d averaged array
 * \param num_rows Number of rows in `input`
 * \param num_cols Number of columns in `input`
 */

void average_col_major_2d(std::vector<double>& input, std::vector<double>& output, int num_rows, int num_cols) {
  for (int i = 0; i < num_rows; i++) {
    double sum = 0;
    for (int j = 0; j < num_cols; j++) {
      sum += input[j * num_rows + i];
    }
    output[i] = sum / num_cols;
  }
}

/*!
 * \brief Internal helper function to average over a specific axis of a column-major 3d array.
 * input_idx_scale1, input_idx_scale2, input_idx_scale3 refer to the scaling factors for the three dimensions of the input array, which are used to compute the correct index in the flattened input vector.
 * output_idx_scale1 and output_idx_scale2 refer to the scaling factors for the two dimensions of the output array, which are used to compute the correct index in the flattened output vector.
 *
 * \param input Vector representation of a 3d array stored in column-major order
 * \param output Empty vector allocated to store the 2d averaged array in column-major order
 * \param dim1 Size of the first dimension of the loop (NOTE: ~not~ necessarily the first dimension of the array)
 * \param dim2 Size of the second dimension of the loop (NOTE: ~not~ necessarily the second dimension of the array)
 * \param dim3 Size of the third dimension of the loop (NOTE: ~not~ necessarily the third dimension of the array)
 * \param input_idx_scale1 Scaling factor for the index corresponding to the first dimension of the loop, which ensures that the correct array index is computed when accessing data from `input`
 * \param input_idx_scale2 Scaling factor for the index corresponding to the second dimension of the loop, which ensures that the correct array index is computed when accessing data from `input`
 * \param input_idx_scale3 Scaling factor for the index corresponding to the third dimension of the loop, which ensures that the correct array index is computed when accessing data from `input`
 * \param output_idx_scale1 Scaling factor for the index corresponding to the first dimension of the loop, which ensures that the correct array index is computed when storing data in `output`
 * \param output_idx_scale2 Scaling factor for the index corresponding to the second dimension of the loop, which ensures that the correct array index is computed when storing data in `output`
 */
void _average_col_major_3d_helper(std::vector<double>& input, std::vector<double>& output, int dim1, int dim2, int dim3,
                                  int input_idx_scale1, int input_idx_scale2, int input_idx_scale3,
                                  int output_idx_scale1, int output_idx_scale2) {
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      double sum = 0;
      for (int k = 0; k < dim3; k++) {
        const int input_idx = k * input_idx_scale3 + j * input_idx_scale2 + i * input_idx_scale1;
        sum += input[input_idx];
      }
      const int output_idx = j * output_idx_scale2 + i * output_idx_scale1;
      output[output_idx] = sum / dim3;
    }
  }
}

/*!
 * \brief Internal helper function to average over a specific axis of a column-major 3d array. Works similarly to `np.mean(..., axis=dim_average)` in numpy. The `dim_average` parameter specifies which dimension to average over.
 * Assumes `input` has size dim1 x dim2 x dim3 and is stored in column-major order and that `output` has size corresponding to the two dimensions that are not being averaged over (i.e. if dim_average==0, output should have size dim2 x dim3; if dim_average==1, output should have size dim1 x dim3; if dim_average==2, output should have size dim1 x dim2).
 * input_idx_scale1, input_idx_scale2, input_idx_scale3 refer to the scaling factors for the three dimensions of the input array, which are used to compute the correct index in the flattened input vector.
 *
 * \param input Vector representation of a 3d array stored in column-major order
 * \param output Empty vector allocated to store the 2d averaged array in column-major order
 * \param dim1 Size of the first dimension of the loop (NOTE: ~not~ necessarily the first dimension of the array)
 * \param dim2 Size of the second dimension of the loop (NOTE: ~not~ necessarily the second dimension of the array)
 * \param dim3 Size of the third dimension of the loop (NOTE: ~not~ necessarily the third dimension of the array)
 * \param dim_average Dimension of the input array over which to average (must be 0, 1, or 2)
 */
void average_col_major_3d(std::vector<double>& input, std::vector<double>& output, int dim1, int dim2, int dim3, int dim_average) {
  if (dim_average == 2) {
    _average_col_major_3d_helper(input, output, dim1, dim2, dim3, 1, dim1, dim1 * dim2, 1, dim1);
  } else if (dim_average == 1) {
    _average_col_major_3d_helper(input, output, dim1, dim3, dim2, 1, dim1 * dim2, dim1, 1, dim1);
  } else if (dim_average == 0) {
    _average_col_major_3d_helper(input, output, dim2, dim3, dim1, dim1, dim1 * dim2, 1, 1, dim2);
  } else {
    Log::Fatal("dim_average must be in {0, 1, 2}.");
  }
}

/*!
 * The return value, BARTPredictionResult, is a struct that contains many optional data fields
 * stored as std::vectors that are left empty if a model term is not requested by the prediction call.
 *
 * In some cases, model terms need to be computed even if not directly requested.
 * For example, the conditional outcome mean (y_hat) requires mean forest and any random effects predictions.
 * In the case that a term is needed as an intermediate computation but not requested as an output, we
 * compute it internally and not return it.
 */
BARTPredictionResult predict_bart_model(BARTData& data, BARTPredictionInput& model_refs) {
  // Initialize a prediction result object
  BARTPredictionResult output{};

  // Key input / output dimensions
  const int num_samples = model_refs.num_samples;
  const int num_obs = model_refs.num_obs;
  // const int num_basis = model_refs.num_basis;

  // Key model components
  const bool has_mean_forest = model_refs.mean_forests != nullptr;
  const bool has_variance_forest = model_refs.has_variance_forest;
  const bool has_rfx = model_refs.has_rfx;
  const bool rfx_custom = model_refs.rfx_model_spec == BARTRFXModelSpec::Custom;
  // const bool rfx_intercept = model_refs.rfx_model_spec == BARTRFXModelSpec::InterceptOnly;

  // Input data / config checks
  if (has_rfx) {
    if (rfx_custom && data.rfx_basis_test == nullptr) {
      Log::Fatal("Model includes random effects with custom basis, but no random effect basis was provided in the test data for prediction");
    }
  }
  if (model_refs.pred_scale == PredScale::kClass && model_refs.pred_type == PredType::kMean) {
    Log::Fatal("Taking the posterior mean of integer-valued class predictions is not an informative quantity, so this combination of pred_scale and pred_type is not supported directly by stochtree's prediction capabilities. If you do wish to obtain a posterior mean of class label predictions, we recommend predicting the class label posterior and then taking the average across MCMC samples in the resulting array");
  }

  // Model output details:
  // - num_samples_output refers to the posterior sample dimension, which is num_samples for posterior predictions and 1 for posterior mean transformations
  // - each of the need_* fields are true if a term needs to be computed en route to the user's requested outputs
  int num_samples_output = 1;
  if (model_refs.pred_type == PredType::kPosterior) {
    num_samples_output = num_samples;
  }
  bool need_mean = has_mean_forest && (model_refs.pred_terms.y_hat || model_refs.pred_terms.mean_forest);
  bool need_rfx = has_rfx && (model_refs.pred_terms.y_hat || model_refs.pred_terms.random_effects);
  bool need_variance_forest = has_variance_forest && model_refs.pred_terms.variance_forest;

  // Resize any output vectors to be returned to users
  const bool probability_scale = model_refs.pred_scale == PredScale::kProbability;
  const bool class_scale = model_refs.pred_scale == PredScale::kClass;
  const bool ordinal_cloglog_prob_scale = probability_scale && model_refs.link_function == LinkFunction::Cloglog && model_refs.outcome_type == OutcomeType::Ordinal;
  if (model_refs.pred_terms.y_hat) output.y_hat.resize(num_obs * (ordinal_cloglog_prob_scale ? model_refs.cloglog_num_classes : 1) * num_samples_output);
  if (model_refs.pred_terms.mean_forest) output.mean_forest_predictions.resize(num_obs * (ordinal_cloglog_prob_scale ? model_refs.cloglog_num_classes : 1) * num_samples_output);
  if (model_refs.pred_terms.variance_forest) output.variance_forest_predictions.resize(num_obs * num_samples_output);
  if (model_refs.pred_terms.random_effects) output.rfx_predictions.resize(num_obs * num_samples_output);

  // Initialize temporary containers needed to compute the requested predictions
  std::vector<double> mean_forest;
  std::vector<double> rfx;
  std::vector<double> variance_forest;
  std::vector<double> y_hat;
  if (need_mean) {
    mean_forest.resize(num_obs * num_samples);
  }
  if (need_rfx) {
    rfx.resize(num_obs * num_samples);
  }
  if (need_variance_forest) {
    variance_forest.resize(num_obs * num_samples);
  }
  if (model_refs.pred_terms.y_hat) {
    y_hat.resize(num_obs * num_samples);
  }

  // Construct ForestDataset -- use the "test" fields
  ForestDataset forest_dataset{};
  forest_dataset.AddCovariates(data.X_test, data.n_test, data.p, /*row_major=*/false);
  if (data.basis_test != nullptr) {
    forest_dataset.AddBasis(data.basis_test, data.n_test, data.basis_dim, /*row_major=*/false);
  }

  if (need_mean) {
    // Predict from mean forest
    mean_forest = model_refs.mean_forests->Predict(forest_dataset);
  }

  // Compute overall random effects predictions
  if (need_rfx) {
    RandomEffectsDataset rfx_dataset;
    rfx_dataset.AddGroupLabels(data.rfx_group_ids_test, num_obs);
    if (data.rfx_basis_test != nullptr) {
      rfx_dataset.AddBasis(data.rfx_basis_test, num_obs, data.rfx_basis_dim, /*row_major=*/false);
    } else if (model_refs.rfx_model_spec == BARTRFXModelSpec::InterceptOnly) {
      std::vector<double> rfx_basis(data.n_test, 1.0);
      rfx_dataset.AddBasis(rfx_basis.data(), num_obs, 1, /*row_major=*/false);
    } else {
      Log::Fatal("BART model random effects term was not sampled with intercept_only or intercept_plus_treatment specification, but not random effect basis was provided for prediction");
    }
    model_refs.rfx_container->Predict(rfx_dataset, *model_refs.rfx_label_mapper, rfx);
  }

  if (need_variance_forest) {
    variance_forest = model_refs.variance_forests->Predict(forest_dataset);
  }
  if (model_refs.pred_terms.y_hat) {
    // y_hat is default initialized to 0, so we can just add the mean forest and random effects predictions as needed
    for (int i = 0; i < num_obs; i++) {
      if (need_mean) {
        for (int j = 0; j < num_samples; j++) {
          y_hat[j * num_obs + i] += mean_forest[j * num_obs + i];
        }
      }
      if (need_rfx) {
        for (int j = 0; j < num_samples; j++) {
          y_hat[j * num_obs + i] += rfx[j * num_obs + i];
        }
      }
    }
  }

  // Scale the outputs
  if (model_refs.pred_terms.mean_forest) {
    location_scale_adjust_predictions(mean_forest, model_refs.y_bar, model_refs.y_std);
  }
  if (model_refs.pred_terms.random_effects) {
    location_scale_adjust_predictions(rfx, 0.0, model_refs.y_std);
  }
  if (model_refs.pred_terms.y_hat) {
    location_scale_adjust_predictions(y_hat, model_refs.y_bar, model_refs.y_std);
  }
  if (need_variance_forest) {
    location_scale_adjust_predictions(variance_forest, 0.0, model_refs.y_std * model_refs.y_std);
  }

  // Transform if necessary (e.g. for probit models)
  if (model_refs.link_function == LinkFunction::Probit) {
    if (model_refs.pred_terms.mean_forest && probability_scale) {
      probability_transform_probit_predictions(mean_forest);
    }
    if (model_refs.pred_terms.random_effects && probability_scale) {
      probability_transform_probit_predictions(rfx);
    }
    if (model_refs.pred_terms.y_hat && (probability_scale || class_scale)) {
      probability_transform_probit_predictions(y_hat);
      if (class_scale) {
        class_transform_binary_outcome_predictions(y_hat);
      }
    }
  } else if (model_refs.link_function == LinkFunction::Cloglog) {
    if (model_refs.outcome_type == OutcomeType::Binary) {
      if (model_refs.pred_terms.mean_forest && probability_scale) {
        probability_transform_binary_cloglog_predictions(mean_forest);
      }
      // NOTE: RFX not compatible with cloglog link, so we skip RFX transformation
      if (model_refs.pred_terms.y_hat && (probability_scale || class_scale)) {
        probability_transform_binary_cloglog_predictions(y_hat);
        if (class_scale) {
          class_transform_binary_outcome_predictions(y_hat);
        }
      }
    } else if (model_refs.outcome_type == OutcomeType::Ordinal) {
      if (model_refs.pred_terms.mean_forest && probability_scale) {
        std::vector<double> mean_forest_prob(num_obs * num_samples * model_refs.cloglog_num_classes);
        probability_transform_ordinal_cloglog_predictions(mean_forest, mean_forest_prob, model_refs.cloglog_cutpoint_samples, num_obs, model_refs.cloglog_num_classes, num_samples);
        mean_forest = std::move(mean_forest_prob);
      }
      if (model_refs.pred_terms.y_hat && (probability_scale || class_scale)) {
        std::vector<double> y_hat_prob(num_obs * num_samples * model_refs.cloglog_num_classes);
        probability_transform_ordinal_cloglog_predictions(y_hat, y_hat_prob, model_refs.cloglog_cutpoint_samples, num_obs, model_refs.cloglog_num_classes, num_samples);
        if (model_refs.pred_scale == PredScale::kClass) {
          class_transform_multiclass_outcome_predictions(y_hat_prob, y_hat, num_obs, model_refs.cloglog_num_classes, num_samples);
        } else {
          y_hat = std::move(y_hat_prob);
        }
      }
    }
  }

  // Unpack into returned outputs, aggregating if necessary
  if (model_refs.pred_terms.mean_forest) {
    if (model_refs.pred_type == PredType::kMean) {
      if (model_refs.pred_scale == PredScale::kProbability && model_refs.outcome_type == OutcomeType::Ordinal && model_refs.link_function == LinkFunction::Cloglog) {
        average_col_major_3d(mean_forest, output.mean_forest_predictions, /*dim1=*/num_obs, /*dim2=*/model_refs.cloglog_num_classes, /*dim3=*/num_samples, /*dim_average=*/2);
      } else {
        average_col_major_2d(mean_forest, output.mean_forest_predictions, /*num_rows=*/num_obs, /*num_cols=*/num_samples);
      }
    } else {
      output.mean_forest_predictions = std::move(mean_forest);
    }
  }
  if (need_variance_forest) {
    if (model_refs.pred_type == PredType::kMean) {
      // NOTE: variance forest not compatible with ordinal cloglog model so we don't need to worry about 3d averaging here
      average_col_major_2d(variance_forest, output.variance_forest_predictions, /*num_rows=*/num_obs, /*num_cols=*/num_samples);
    } else {
      output.variance_forest_predictions = std::move(variance_forest);
    }
  }
  if (model_refs.pred_terms.random_effects) {
    if (model_refs.pred_type == PredType::kMean) {
      // NOTE: random effects not compatible with ordinal cloglog model so we don't need to worry about 3d averaging here
      average_col_major_2d(rfx, output.rfx_predictions, /*num_rows=*/num_obs, /*num_cols=*/num_samples);
    } else {
      output.rfx_predictions = std::move(rfx);
    }
  }
  if (model_refs.pred_terms.y_hat) {
    if (model_refs.pred_type == PredType::kMean) {
      if (model_refs.pred_scale == PredScale::kProbability && model_refs.outcome_type == OutcomeType::Ordinal && model_refs.link_function == LinkFunction::Cloglog) {
        average_col_major_3d(y_hat, output.y_hat, /*dim1=*/num_obs, /*dim2=*/model_refs.cloglog_num_classes, /*dim3=*/num_samples, /*dim_average=*/2);
      } else {
        average_col_major_2d(y_hat, output.y_hat, /*num_rows=*/num_obs, /*num_cols=*/num_samples);
      }
    } else {
      output.y_hat = std::move(y_hat);
    }
  }

  return output;
}

/*!
 * The return value, BCFPRedictionResult, is a struct that contains many optional data fields
 * stored as std::vectors that are left empty if a model term is not requested by the prediction call.
 *
 * In some cases, model terms need to be computed even if not directly requested.
 * For example, the conditional outcome mean (y_hat) requires mu_x, tau_x and any random effects predictions.
 * In the case that a term is needed as an intermediate computation but not requested as an output, we
 * compute it internally and not return it.
 */
BCFPredictionResult predict_bcf_model(BCFData& data, BCFPredictionInput& model_refs) {
  // Initialize a prediction result object
  BCFPredictionResult output{};

  // Key input / output dimensions
  const int num_samples = model_refs.num_samples;
  const int num_obs = model_refs.num_obs;
  const int num_treatment = model_refs.treatment_dim;

  // Key model components
  const bool has_mu_forest = model_refs.mu_forests != nullptr;
  const bool has_tau_forest = model_refs.tau_forests != nullptr;
  const bool has_variance_forest = model_refs.variance_forests != nullptr;
  const bool has_rfx = model_refs.rfx_container != nullptr;
  const bool rfx_custom = model_refs.rfx_model_spec == BCFRFXModelSpec::Custom;
  const bool rfx_intercept = model_refs.rfx_model_spec == BCFRFXModelSpec::InterceptOnly || model_refs.rfx_model_spec == BCFRFXModelSpec::InterceptPlusTreatment;
  const bool rfx_treatment = model_refs.rfx_model_spec == BCFRFXModelSpec::InterceptPlusTreatment;

  // Model output details:
  // - num_samples_output refers to the posterior sample dimension, which is num_samples for posterior predictions and 1 for posterior mean transformations
  // - each of the need_* fields are true if a term needs to be computed en route to the user's requested outputs
  int num_samples_output = 1;
  if (model_refs.pred_type == PredType::kPosterior) {
    num_samples_output = num_samples;
  }
  bool need_tau_interm = model_refs.pred_terms.y_hat || model_refs.pred_terms.tau_x || model_refs.pred_terms.cate;
  bool need_mu = model_refs.pred_terms.y_hat || model_refs.pred_terms.mu_x || model_refs.pred_terms.prognostic_function || (model_refs.adaptive_coding && need_tau_interm);
  bool need_tau = need_tau_interm || (model_refs.adaptive_coding && need_mu);
  bool need_rfx = has_rfx && (model_refs.pred_terms.y_hat || model_refs.pred_terms.random_effects);
  bool need_rfx_intercept = has_rfx && rfx_intercept && model_refs.pred_terms.prognostic_function;
  bool need_rfx_treatment = has_rfx && rfx_treatment && model_refs.pred_terms.cate;
  bool need_variance_forest = has_variance_forest && model_refs.pred_terms.conditional_variance;

  // Resize any output vectors to be returned to users
  if (model_refs.pred_terms.y_hat) output.y_hat.resize(num_obs * num_samples_output);
  if (model_refs.pred_terms.mu_x) output.mu_x.resize(num_obs * num_samples_output);
  if (model_refs.pred_terms.tau_x) output.tau_x.resize(num_obs * num_treatment * num_samples_output);
  if (model_refs.pred_terms.prognostic_function) output.prognostic_function.resize(num_obs * num_samples_output);
  if (model_refs.pred_terms.cate) output.cate.resize(num_obs * num_treatment * num_samples_output);
  if (model_refs.pred_terms.conditional_variance) output.conditional_variance.resize(num_obs * num_samples_output);
  if (model_refs.pred_terms.random_effects) output.random_effects.resize(num_obs * num_samples_output);

  // Initialize temporary containers needed to compute the requested predictions
  std::vector<double> mu_x;
  std::vector<double> prognostic_function;
  std::vector<double> tau_x;
  std::vector<double> cate;
  std::vector<double> rfx_mu;
  std::vector<double> rfx_tau;
  std::vector<double> rfx;
  std::vector<double> variance_forest;
  std::vector<double> y_hat;
  if (need_mu) {
    mu_x.resize(num_obs * num_samples);
  }
  if (model_refs.pred_terms.prognostic_function) {
    prognostic_function.resize(num_obs * num_samples);
  }
  if (need_tau) {
    tau_x.resize(num_obs * num_treatment * num_samples);
  }
  if (model_refs.pred_terms.cate) {
    cate.resize(num_obs * num_treatment * num_samples);
  }
  if (need_rfx_intercept) {
    rfx_mu.resize(num_obs * num_samples);
  }
  if (need_rfx_treatment) {
    rfx_tau.resize(num_obs * num_treatment * num_samples);
  }
  if (need_rfx) {
    rfx.resize(num_obs * num_samples);
  }
  if (need_variance_forest) {
    variance_forest.resize(num_obs * num_samples);
  }
  if (model_refs.pred_terms.y_hat) {
    y_hat.resize(num_obs * num_samples);
  }

  // Construct ForestDataset -- use the "test" fields
  ForestDataset forest_dataset{};
  forest_dataset.AddCovariates(data.X_test, data.n_test, data.p, /*row_major=*/false);
  // NOTE: not adding treatment as basis to forest_dataset since we always predict the raw treatment effect forest values and multiply by
  // either the raw or recoded treatment (if adaptive coding)
  if (data.obs_weights_test != nullptr) {
    forest_dataset.AddVarianceWeights(data.obs_weights_test, data.n_test);
  }

  if (need_mu) {
    // Predict from mu forest
    mu_x = model_refs.mu_forests->Predict(forest_dataset);
  }

  if (need_tau) {
    // Predict from tau forest. We use PredictRaw for the tau forest because we
    // don't want to pre-multiply by the treatment / basis at this stage -- we want to be
    // able to return the treatment effect itself, not the treatment effect times Z (or recoded Z)
    tau_x = model_refs.tau_forests->PredictRaw(forest_dataset, /*row_major=*/false);
    // Add tau_0 to the treatment effect function predictions if it was sampled.
    // tau_0_samples layout: col-major (treatment dim k, sample j) -> j * treatment_dim + k.
    // For treatment_dim==1 this collapses to samples.tau_0_samples[j].
    // NOTE: tau_0_samples is stored in original (unstandardized) scale; tau_x from PredictRaw
    // is in standardized scale. Divide by y_std to convert tau_0 to standardized scale
    // before adding, so the y_std scale step applied later gives the right result.
    if (model_refs.sample_tau_0) {
      const double inv_y_std = 1.0 / model_refs.y_std;
      for (int j = 0; j < num_samples; j++) {
        for (int k = 0; k < num_treatment; k++) {
          for (int i = 0; i < num_obs; i++) {
            const int idx = j * num_obs * num_treatment + k * num_obs + i;
            tau_x[idx] += model_refs.tau_0_samples[j * num_treatment + k] * inv_y_std;
          }
        }
      }
    }
    // Handle adaptive coding correctly:
    // When treatment is b_0 (1-Z) + b_1 Z, the conditional mean model:
    //      mu(x) + [tau_0 + tau(x)] * (b_0 * (1-Z) + b_1 * Z)
    // turns into
    //      [mu(x) + b_0 * (tau_0 + tau(x))] + (tau_0 + tau(x)) * (b_1 - b_0) * Z
    // So the treatment effect function that gets multiplied by Z is actually (b_1 - b_0) * (tau_0 + tau(x))
    // and the prognostic function has an added contribution of b_0 * (tau_0 + tau(x))
    // NOTE: adaptive coding is only supported for a univariate binary treatment, so we construct our indices as if tau_x is 2d because whenever adaptive coding is true, treatment_dim must be 1 and the array is effectively 2d.
    if (model_refs.adaptive_coding) {
      for (int i = 0; i < num_samples; i++) {
        double b_0 = model_refs.b0_samples[i];
        double b_1 = model_refs.b1_samples[i];
        for (int j = 0; j < num_obs; j++) {
          const int idx = i * num_obs + j;
          // Add b_0 * (tau_0 + tau(x)) to the prognostic function predictions
          mu_x[idx] += b_0 * tau_x[idx];
          // Scale tau_x by (b_1 - b_0)
          tau_x[idx] *= (b_1 - b_0);
        }
      }
    }
  }

  // Add random effects contribution to prognostic function if needed
  if (need_rfx_intercept) {
    // Extract just the random intercept effects from the RFX model
    int group_ind;
    const int k = 0;  // We only want the first column from the RFX parameters
    std::vector<double>& beta = model_refs.rfx_container->GetBeta();
    int num_components = model_refs.rfx_container->NumComponents();
    int num_groups = model_refs.rfx_container->NumGroups();
    for (int i = 0; i < num_obs; i++) {
      group_ind = model_refs.rfx_label_mapper->CategoryNumber(data.rfx_group_ids_test[i]);
      for (int j = 0; j < num_samples; j++) {
        const int idx = j * num_obs + i;
        rfx_mu[idx] = beta.at(j * num_groups * num_components + group_ind * num_components + k);
      }
    }
  }

  // Add random effects contribution to CATE function if needed
  if (need_rfx_treatment) {
    // Extract just the random treatment effects from the RFX model
    int group_ind;
    std::vector<double>& beta = model_refs.rfx_container->GetBeta();
    int num_components = model_refs.rfx_container->NumComponents();
    int num_groups = model_refs.rfx_container->NumGroups();
    for (int i = 0; i < num_obs; i++) {
      group_ind = model_refs.rfx_label_mapper->CategoryNumber(data.rfx_group_ids_test[i]);
      for (int j = 0; j < num_samples; j++) {
        // In the "intercept_plus_treatment" RFX specification, the random intercept is in column 0 and the random treatment effect(s) start from column 1,
        // so we loop from k=1 to num_components to extract the treatment effect contribution(s)
        for (int k = 1; k < num_components; k++) {
          // NOTE, the "treatment" index is k - 1 since k starts at 1 in this loop
          const int idx = j * num_obs * num_treatment + (k - 1) * num_obs + i;
          rfx_tau[idx] = beta.at(j * num_groups * num_components + group_ind * num_components + k);
        }
      }
    }
  }

  // Compute overall random effects predictions
  if (need_rfx) {
    RandomEffectsDataset rfx_dataset;
    rfx_dataset.AddGroupLabels(data.rfx_group_ids_test, num_obs);
    if (data.rfx_basis_test != nullptr) {
      rfx_dataset.AddBasis(data.rfx_basis_test, num_obs, data.rfx_basis_dim, /*row_major=*/false);
    } else if (model_refs.rfx_model_spec == BCFRFXModelSpec::InterceptOnly) {
      std::vector<double> rfx_basis(data.n_test, 1.0);
      rfx_dataset.AddBasis(rfx_basis.data(), num_obs, 1, /*row_major=*/false);
    } else if (model_refs.rfx_model_spec == BCFRFXModelSpec::InterceptPlusTreatment) {
      // Column-major rfx basis
      std::vector<double> rfx_basis(data.n_test * (1 + num_treatment));
      for (int i = 0; i < num_obs; i++) {
        rfx_basis[i] = 1.0;
      }
      for (int j = 0; j < num_treatment; j++) {
        for (int i = 0; i < num_obs; i++) {
          rfx_basis[(j + 1) * num_obs + i] = data.treatment_test[j * num_obs + i];
        }
      }
      rfx_dataset.AddBasis(rfx_basis.data(), num_obs, 1 + num_treatment, /*row_major=*/false);
    } else {
      Log::Fatal("BCF model random effects term was not sampled with intercept_only or intercept_plus_treatment specification, but not random effect basis was provided for prediction");
    }
    model_refs.rfx_container->Predict(rfx_dataset, *model_refs.rfx_label_mapper, rfx);
  }

  // Unpack into returned outputs
  if (model_refs.pred_terms.prognostic_function) {
    for (int i = 0; i < mu_x.size(); i++) {
      prognostic_function[i] = mu_x[i];
      if (need_rfx_intercept) {
        prognostic_function[i] += rfx_mu[i];
      }
    }
  }
  if (model_refs.pred_terms.cate) {
    for (int i = 0; i < tau_x.size(); i++) {
      cate[i] = tau_x[i];
      if (need_rfx_treatment) {
        cate[i] += rfx_tau[i];
      }
    }
  }
  if (need_variance_forest) {
    variance_forest = model_refs.variance_forests->Predict(forest_dataset);
  }
  if (model_refs.pred_terms.y_hat) {
    for (int i = 0; i < num_obs; i++) {
      for (int j = 0; j < num_samples; j++) {
        y_hat[j * num_obs + i] = mu_x[j * num_obs + i];
        for (int k = 0; k < num_treatment; k++) {
          y_hat[j * num_obs + i] += tau_x[j * num_obs * num_treatment + k * num_obs + i] * data.treatment_test[k * num_obs + i];
        }
      }
      if (need_rfx) {
        for (int j = 0; j < num_samples; j++) {
          y_hat[j * num_obs + i] += rfx[j * num_obs + i];
        }
      }
    }
  }

  // Scale the outputs
  if (model_refs.pred_terms.mu_x) {
    location_scale_adjust_predictions(mu_x, model_refs.y_bar, model_refs.y_std);
  }
  if (model_refs.pred_terms.prognostic_function) {
    location_scale_adjust_predictions(prognostic_function, model_refs.y_bar, model_refs.y_std);
  }
  if (model_refs.pred_terms.tau_x) {
    location_scale_adjust_predictions(tau_x, 0.0, model_refs.y_std);
  }
  if (model_refs.pred_terms.cate) {
    location_scale_adjust_predictions(cate, 0.0, model_refs.y_std);
  }
  if (model_refs.pred_terms.random_effects) {
    location_scale_adjust_predictions(rfx, 0.0, model_refs.y_std);
  }
  if (model_refs.pred_terms.y_hat) {
    location_scale_adjust_predictions(y_hat, model_refs.y_bar, model_refs.y_std);
  }
  if (need_variance_forest) {
    location_scale_adjust_predictions(variance_forest, 0.0, model_refs.y_std * model_refs.y_std);
  }

  // Transform if necessary (e.g. for probit models)
  // NOTE: if we support cloglog or ordinal probit BCF in the future (likely),
  // we must add more link function guards to this block of code
  const bool probability_scale = model_refs.pred_scale == PredScale::kProbability;
  const bool class_scale = model_refs.pred_scale == PredScale::kClass;
  if (model_refs.pred_terms.mu_x && probability_scale) {
    probability_transform_probit_predictions(mu_x);
  }
  if (model_refs.pred_terms.prognostic_function && probability_scale) {
    probability_transform_probit_predictions(prognostic_function);
  }
  if (model_refs.pred_terms.tau_x && probability_scale) {
    probability_transform_probit_predictions(tau_x);
  }
  if (model_refs.pred_terms.cate && probability_scale) {
    probability_transform_probit_predictions(cate);
  }
  if (model_refs.pred_terms.y_hat && (probability_scale || class_scale)) {
    probability_transform_probit_predictions(y_hat);
    if (class_scale) {
      class_transform_binary_outcome_predictions(y_hat);
    }
  }

  // Unpack into returned outputs, aggregating if necessary
  if (model_refs.pred_terms.mu_x) {
    if (model_refs.pred_type == PredType::kMean) {
      average_col_major_2d(mu_x, output.mu_x, num_obs, num_samples);
    } else {
      output.mu_x = std::move(mu_x);
    }
  }
  if (model_refs.pred_terms.tau_x) {
    if (model_refs.pred_type == PredType::kMean) {
      if (num_treatment == 1) {
        // If only one treatment, tau_x is num_obs by num_samples, so average across samples in columns
        average_col_major_2d(tau_x, output.tau_x, /*num_rows=*/num_obs, /*num_cols=*/num_samples);
      } else {
        // If multiple treatments, tau_x is num_obs by num_treatment by num_samples in column-major order, so average across samples while keeping treatment dimension separate
        average_col_major_3d(tau_x, output.tau_x, /*dim1=*/num_obs, /*dim2=*/num_treatment, /*dim3=*/num_samples, /*dim_average=*/2);
      }
    } else {
      output.tau_x = std::move(tau_x);
    }
  }
  if (model_refs.pred_terms.prognostic_function) {
    if (model_refs.pred_type == PredType::kMean) {
      average_col_major_2d(prognostic_function, output.prognostic_function, num_obs, num_samples);
    } else {
      output.prognostic_function = std::move(prognostic_function);
    }
  }
  if (model_refs.pred_terms.cate) {
    if (model_refs.pred_type == PredType::kMean) {
      if (num_treatment == 1) {
        // If only one treatment, cate is num_obs by num_samples, so average across samples in columns
        average_col_major_2d(cate, output.cate, /*num_rows=*/num_obs, /*num_cols=*/num_samples);
      } else {
        // If multiple treatments, cate is num_obs by num_treatment by num_samples in column-major order, so average across samples while keeping treatment dimension separate
        average_col_major_3d(cate, output.cate, /*dim1=*/num_obs, /*dim2=*/num_treatment, /*dim3=*/num_samples, /*dim_average=*/2);
      }
    } else {
      output.cate = std::move(cate);
    }
  }
  if (need_variance_forest) {
    if (model_refs.pred_type == PredType::kMean) {
      average_col_major_2d(variance_forest, output.conditional_variance, num_obs, num_samples);
    } else {
      output.conditional_variance = std::move(variance_forest);
    }
  }
  if (model_refs.pred_terms.random_effects) {
    if (model_refs.pred_type == PredType::kMean) {
      average_col_major_2d(rfx, output.random_effects, num_obs, num_samples);
    } else {
      output.random_effects = std::move(rfx);
    }
  }
  if (model_refs.pred_terms.y_hat) {
    if (model_refs.pred_type == PredType::kMean) {
      average_col_major_2d(y_hat, output.y_hat, num_obs, num_samples);
    } else {
      output.y_hat = std::move(y_hat);
    }
  }

  return output;
}

}  // namespace StochTree