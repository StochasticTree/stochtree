# R6 wrapper around the C++ BARTSamples object

#' @description
#' Container holding a sampled BART model's forests and parameter traces as a single C++ object.
#' @noRd
BARTSamples <- R6::R6Class(
  classname = "BARTSamples",
  cloneable = FALSE,
  public = list(
    #' @field samples_ptr External pointer to a C++ BARTSamples object
    samples_ptr = NULL,

    #' @description
    #' Initialize an empty BARTSamples object in C++ and wrap an external pointer to the object.
    initialize = function() {
      self$samples_ptr <- bart_samples_cpp()
    },

    #' @description
    #' Initialize a BARTSamples object from JSON and wrap an external pointer to the object.
    from_json = function(json) {
      self$samples_ptr <- bart_samples_from_json_cpp(json$json_ptr)
    },

    #' @description
    #' Convert a BARTSamples object to JSON and include it in a `CppJson` object wrapping a C++ JSON representation.
    append_to_json = function(json) {
      append_bart_samples_to_json_cpp(self$samples_ptr, json$json_ptr)
    },

    #' @description Number of retained posterior samples.
    num_samples = function() bart_samples_num_samples_cpp(self$samples_ptr),

    #' @description Outcome mean used for standardization.
    y_bar = function() bart_samples_y_bar_cpp(self$samples_ptr),

    #' @description Outcome standard deviation used for standardization.
    y_std = function() bart_samples_y_std_cpp(self$samples_ptr),

    #' @description Whether global error scale samples are present.
    has_global_var_samples = function() {
      bart_samples_has_global_var_samples_cpp(self$samples_ptr)
    },

    #' @description Whether leaf scale samples are present.
    has_leaf_scale_samples = function() {
      bart_samples_has_leaf_scale_samples_cpp(self$samples_ptr)
    },

    #' @description Whether outcome predictions for the training set are present.
    has_yhat_train = function() {
      bart_samples_has_yhat_train_cpp(self$samples_ptr)
    },

    #' @description Whether mean forest predictions for the training set are present.
    has_mean_forest_predictions_train = function() {
      bart_samples_has_mean_forest_predictions_train_cpp(self$samples_ptr)
    },

    #' @description Whether variance forest predictions for the training set are present.
    has_variance_forest_predictions_train = function() {
      bart_samples_has_variance_forest_predictions_train_cpp(self$samples_ptr)
    },

    #' @description Whether outcome predictions for the test set are present.
    has_yhat_test = function() {
      bart_samples_has_yhat_test_cpp(self$samples_ptr)
    },

    #' @description Whether mean forest predictions for the test set are present.
    has_mean_forest_predictions_test = function() {
      bart_samples_has_mean_forest_predictions_test_cpp(self$samples_ptr)
    },

    #' @description Whether variance forest predictions for the test set are present.
    has_variance_forest_predictions_test = function() {
      bart_samples_has_variance_forest_predictions_test_cpp(self$samples_ptr)
    },

    #' @description Whether a mean forest is present.
    has_mean_forest = function() {
      bart_samples_has_mean_forest_cpp(self$samples_ptr)
    },

    #' @description Whether a variance forest is present.
    has_variance_forest = function() {
      bart_samples_has_variance_forest_cpp(self$samples_ptr)
    },

    #' @description Whether random effects are present.
    has_rfx = function() {
      bart_samples_has_rfx_cpp(self$samples_ptr)
    },

    #' @description Global error variance samples (length `num_samples`, or empty).
    global_var_samples = function() {
      bart_samples_global_var_samples_cpp(self$samples_ptr)
    },

    #' @description Leaf scale samples (length `num_samples`, or empty).
    leaf_scale_samples = function() {
      bart_samples_leaf_scale_samples_cpp(self$samples_ptr)
    },

    #' @description Whether cloglog cutpoint samples are present.
    has_cloglog_cutpoint_samples = function() {
      bart_samples_has_cloglog_cutpoint_samples_cpp(self$samples_ptr)
    },

    #' @description Cloglog ordinal cutpoint samples (`(num_classes - 1)` x `num_samples`), or NULL.
    cloglog_cutpoint_samples = function() {
      if (!self$has_cloglog_cutpoint_samples()) {
        return(NULL)
      }
      bart_samples_cloglog_cutpoint_samples_cpp(self$samples_ptr)
    },

    #' @description Mean forest predictions for the training set (length `num_samples` * `num_train`, or empty).
    y_hat_train = function() {
      bart_samples_yhat_train_cpp(self$samples_ptr)
    },

    #' @description Mean forest predictions for the training set (length `num_samples` * `num_train`, or empty).
    mean_forest_predictions_train = function() {
      bart_samples_mean_forest_predictions_train_cpp(self$samples_ptr)
    },

    #' @description Variance forest predictions for the training set (length `num_samples` * `num_train`, or empty).
    variance_forest_predictions_train = function() {
      bart_samples_variance_forest_predictions_train_cpp(self$samples_ptr)
    },

    #' @description Random effects predictions for the training set (length `num_samples` * `num_train`, or empty).
    rfx_predictions_train = function() {
      bart_samples_rfx_predictions_train_cpp(self$samples_ptr)
    },

    #' @description Mean forest predictions for the test set (length `num_samples` * `num_test`, or empty).
    y_hat_test = function() {
      bart_samples_yhat_test_cpp(self$samples_ptr)
    },

    #' @description Mean forest predictions for the test set (length `num_samples` * `num_test`, or empty).
    mean_forest_predictions_test = function() {
      bart_samples_mean_forest_predictions_test_cpp(self$samples_ptr)
    },

    #' @description Variance forest predictions for the test set (length `num_samples` * `num_test`, or empty).
    variance_forest_predictions_test = function() {
      bart_samples_variance_forest_predictions_test_cpp(self$samples_ptr)
    },

    #' @description Random effects predictions for the test set (length `num_samples` * `num_test`, or empty).
    rfx_predictions_test = function() {
      bart_samples_rfx_predictions_test_cpp(self$samples_ptr)
    },

    #' @description Materialize a standalone deep copy of the mean forest as a `ForestSamples`
    #' (or NULL if absent).
    materialize_mean_forest = function() {
      if (!self$has_mean_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- bart_samples_materialize_mean_forest_cpp(
        self$samples_ptr
      )
      fc
    },

    #' @description Materialize a standalone deep copy of the variance forest as a `ForestSamples`
    #' (or NULL if absent).
    materialize_variance_forest = function() {
      if (!self$has_variance_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- bart_samples_materialize_variance_forest_cpp(
        self$samples_ptr
      )
      fc
    },

    #' @description Materialize a standalone deep copy of the random effects samples as a `RandomEffectSamples`
    #' (or NULL if absent).
    materialize_rfx = function() {
      if (!self$has_rfx()) {
        return(NULL)
      }
      fc <- RandomEffectSamples$new()
      fc$rfx_container_ptr <- bart_samples_materialize_rfx_container_cpp(
        self$samples_ptr
      )
      fc$label_mapper_ptr <- bart_samples_materialize_rfx_label_mapper_cpp(
        self$samples_ptr
      )
      fc
    },

    #' @description Borrowed (non-owning) external pointer to the mean forest container, for
    #' read-through prediction. Must not outlive this object.
    mean_forest_ptr = function() {
      bart_samples_mean_forest_ptr_cpp(self$samples_ptr)
    },

    #' @description Borrowed (non-owning) external pointer to the variance forest container.
    variance_forest_ptr = function() {
      bart_samples_variance_forest_ptr_cpp(self$samples_ptr)
    },

    #' @description Non-owning `ForestSamples` view over the mean forest (borrowed pointer,
    #' no deep copy) for internal read-only consumers (serialization, kernels). NULL if absent.
    mean_forest_view = function() {
      if (!self$has_mean_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- self$mean_forest_ptr()
      fc
    },

    #' @description Non-owning `ForestSamples` view over the variance forest. NULL if absent.
    variance_forest_view = function() {
      if (!self$has_variance_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- self$variance_forest_ptr()
      fc
    },

    #' @description Append another chain's draws onto this one (multi-chain combine).
    #' @param other Another `BARTSamples` with matching structure/standardization.
    merge = function(other) {
      bart_samples_merge_cpp(self$samples_ptr, other$samples_ptr)
    }
  )
)

#' @description
#' Container holding a sampled BCF model's forests and parameter traces as a single C++ object.
#' @noRd
BCFSamples <- R6::R6Class(
  classname = "BCFSamples",
  cloneable = FALSE,
  public = list(
    #' @field samples_ptr External pointer to a C++ BCFSamples object
    samples_ptr = NULL,

    #' @description
    #' Initialize an empty BCFSamples object in C++ and wrap an external pointer to the object.
    initialize = function() {
      self$samples_ptr <- bcf_samples_cpp()
    },

    #' @description
    #' Initialize a BCFSamples object from JSON and wrap an external pointer to the object.
    from_json = function(json) {
      self$samples_ptr <- bcf_samples_from_json_cpp(json$json_ptr)
    },

    #' @description
    #' Convert a BCFSamples object to JSON and return a `CppJson` object wrapping the C++ JSON representation.
    append_to_json = function(json) {
      append_bcf_samples_to_json_cpp(self$samples_ptr, json$json_ptr)
    },

    #' @description Number of retained posterior samples.
    num_samples = function() bcf_samples_num_samples_cpp(self$samples_ptr),

    #' @description Treatment dimension.
    treatment_dim = function() bcf_samples_treatment_dim_cpp(self$samples_ptr),

    #' @description Outcome mean used for standardization.
    y_bar = function() bcf_samples_y_bar_cpp(self$samples_ptr),

    #' @description Outcome standard deviation used for standardization.
    y_std = function() bcf_samples_y_std_cpp(self$samples_ptr),

    #' @description Whether a prognostic forest is present.
    has_mu_forest = function() bcf_samples_has_mu_forest_cpp(self$samples_ptr),

    #' @description Whether a treatment forest is present.
    has_tau_forest = function() {
      bcf_samples_has_tau_forest_cpp(self$samples_ptr)
    },

    #' @description Whether a variance forest is present.
    has_variance_forest = function() {
      bcf_samples_has_variance_forest_cpp(self$samples_ptr)
    },

    #' @description Whether random effects are present.
    has_rfx = function() {
      bcf_samples_has_rfx_cpp(self$samples_ptr)
    },

    #' @description Global error variance samples.
    global_var_samples = function() {
      bcf_samples_global_var_samples_cpp(self$samples_ptr)
    },

    #' @description Prognostic leaf scale samples.
    leaf_scale_mu_samples = function() {
      bcf_samples_leaf_scale_mu_samples_cpp(self$samples_ptr)
    },

    #' @description Treatment leaf scale samples.
    leaf_scale_tau_samples = function() {
      bcf_samples_leaf_scale_tau_samples_cpp(self$samples_ptr)
    },

    #' @description Adaptive-coding b0 samples.
    b0_samples = function() bcf_samples_b0_samples_cpp(self$samples_ptr),

    #' @description Adaptive-coding b1 samples.
    b1_samples = function() bcf_samples_b1_samples_cpp(self$samples_ptr),

    #' @description Treatment intercept (tau_0) samples (flat).
    tau_0_samples = function() bcf_samples_tau_0_samples_cpp(self$samples_ptr),

    #' @description Mean forest predictions for the training set (length `num_samples` * `num_train`, or empty).
    y_hat_train = function() {
      bcf_samples_yhat_train_cpp(self$samples_ptr)
    },

    #' @description Prognostic forest predictions for the training set (length `num_samples` * `num_train`, or empty).
    mu_forest_predictions_train = function() {
      bcf_samples_mu_forest_predictions_train_cpp(self$samples_ptr)
    },

    #' @description Treatment effect forest predictions for the training set (length `num_samples` * `num_treatment` * `num_train`, or `num_samples` * `num_train` if `num_treatment` <= 1, or empty).
    tau_forest_predictions_train = function() {
      bcf_samples_tau_forest_predictions_train_cpp(self$samples_ptr)
    },

    #' @description Variance forest predictions for the training set (length `num_samples` * `num_train`, or empty).
    variance_forest_predictions_train = function() {
      bcf_samples_variance_forest_predictions_train_cpp(self$samples_ptr)
    },

    #' @description Random effects predictions for the training set (length `num_samples` * `num_train`, or empty).
    rfx_predictions_train = function() {
      bcf_samples_rfx_predictions_train_cpp(self$samples_ptr)
    },

    #' @description Mean forest predictions for the test set (length `num_samples` * `num_test`, or empty).
    y_hat_test = function() {
      bcf_samples_yhat_test_cpp(self$samples_ptr)
    },

    #' @description Prognostic forest predictions for the test set (length `num_samples` * `num_test`, or empty).
    mu_forest_predictions_test = function() {
      bcf_samples_mu_forest_predictions_test_cpp(self$samples_ptr)
    },

    #' @description Treatment effect forest predictions for the test set (length `num_samples` * `num_treatment` * `num_test`, or `num_samples` * `num_test` if `num_treatment` <= 1, or empty).
    tau_forest_predictions_test = function() {
      bcf_samples_tau_forest_predictions_test_cpp(self$samples_ptr)
    },

    #' @description Variance forest predictions for the test set (length `num_samples` * `num_test`, or empty).
    variance_forest_predictions_test = function() {
      bcf_samples_variance_forest_predictions_test_cpp(self$samples_ptr)
    },

    #' @description Random effects predictions for the test set (length `num_samples` * `num_test`, or empty).
    rfx_predictions_test = function() {
      bcf_samples_rfx_predictions_test_cpp(self$samples_ptr)
    },

    #' @description Materialize a deep copy of the prognostic forest as a `ForestSamples`.
    materialize_mu_forest = function() {
      if (!self$has_mu_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- bcf_samples_materialize_mu_forest_cpp(
        self$samples_ptr
      )
      fc
    },

    #' @description Materialize a deep copy of the treatment forest as a `ForestSamples`.
    materialize_tau_forest = function() {
      if (!self$has_tau_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- bcf_samples_materialize_tau_forest_cpp(
        self$samples_ptr
      )
      fc
    },

    #' @description Materialize a deep copy of the variance forest as a `ForestSamples` (or NULL).
    materialize_variance_forest = function() {
      if (!self$has_variance_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- bcf_samples_materialize_variance_forest_cpp(
        self$samples_ptr
      )
      fc
    },

    #' @description Materialize a standalone deep copy of the random effects samples as a `RandomEffectSamples`
    #' (or NULL if absent).
    materialize_rfx = function() {
      if (!self$has_rfx()) {
        return(NULL)
      }
      fc <- RandomEffectSamples$new()
      fc$rfx_container_ptr <- bcf_samples_materialize_rfx_container_cpp(
        self$samples_ptr
      )
      fc$label_mapper_ptr <- bcf_samples_materialize_rfx_label_mapper_cpp(
        self$samples_ptr
      )
      fc
    },

    #' @description Borrowed (non-owning) external pointer to the prognostic forest container.
    mu_forest_ptr = function() bcf_samples_mu_forest_ptr_cpp(self$samples_ptr),

    #' @description Borrowed (non-owning) external pointer to the treatment forest container.
    tau_forest_ptr = function() {
      bcf_samples_tau_forest_ptr_cpp(self$samples_ptr)
    },

    #' @description Borrowed (non-owning) external pointer to the variance forest container.
    variance_forest_ptr = function() {
      bcf_samples_variance_forest_ptr_cpp(self$samples_ptr)
    },

    #' @description Non-owning `ForestSamples` view over the prognostic forest. NULL if absent.
    mu_forest_view = function() {
      if (!self$has_mu_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- self$mu_forest_ptr()
      fc
    },

    #' @description Non-owning `ForestSamples` view over the treatment forest. NULL if absent.
    tau_forest_view = function() {
      if (!self$has_tau_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- self$tau_forest_ptr()
      fc
    },

    #' @description Non-owning `ForestSamples` view over the variance forest. NULL if absent.
    variance_forest_view = function() {
      if (!self$has_variance_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- self$variance_forest_ptr()
      fc
    },

    #' @description Append another chain's draws onto this one (multi-chain combine).
    #' @param other Another `BCFSamples` with matching structure/standardization.
    merge = function(other) {
      bcf_samples_merge_cpp(self$samples_ptr, other$samples_ptr)
    }
  )
)
