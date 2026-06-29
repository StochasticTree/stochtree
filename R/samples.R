# R6 wrapper around the single-owner C++ BARTSamples object, mirroring the ForestSamples idiom
# (R/forest.R): the external pointer lives in a field and methods forward to cpp11 free functions.
# This is the R analog of the Python BARTSamplesCpp wrapper -- one object that owns the sampled
# forests + parameter traces, with materialize-on-demand deep-copied forest views for the
# (deprecated) direct forest accessor.

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
    #' Build a BARTSamples object by deep-copying existing forest containers and parameter arrays.
    #' @param mean_forest `ForestSamples` for the mean forest (or NULL for a variance-only model)
    #' @param variance_forest `ForestSamples` for the variance forest (or NULL)
    #' @param global_var_samples Numeric vector of global error variance samples (or NULL)
    #' @param leaf_scale_samples Numeric vector of leaf scale samples (or NULL)
    #' @param y_bar Outcome mean used for standardization
    #' @param y_std Outcome standard deviation used for standardization
    #' @param num_samples Number of retained posterior samples
    initialize = function(
      mean_forest = NULL,
      variance_forest = NULL,
      global_var_samples = NULL,
      leaf_scale_samples = NULL,
      y_bar = 0.0,
      y_std = 1.0,
      num_samples = 0L
    ) {
      mean_ptr <- if (!is.null(mean_forest)) mean_forest$forest_container_ptr else NULL
      variance_ptr <- if (!is.null(variance_forest)) {
        variance_forest$forest_container_ptr
      } else {
        NULL
      }
      self$samples_ptr <- bart_samples_from_components_cpp(
        mean_ptr,
        variance_ptr,
        global_var_samples,
        leaf_scale_samples,
        y_bar,
        y_std,
        as.integer(num_samples)
      )
    },

    #' @description Number of retained posterior samples.
    num_samples = function() bart_samples_num_samples_cpp(self$samples_ptr),

    #' @description Outcome mean used for standardization.
    y_bar = function() bart_samples_y_bar_cpp(self$samples_ptr),

    #' @description Outcome standard deviation used for standardization.
    y_std = function() bart_samples_y_std_cpp(self$samples_ptr),

    #' @description Whether a mean forest is present.
    has_mean_forest = function() bart_samples_has_mean_forest_cpp(self$samples_ptr),

    #' @description Whether a variance forest is present.
    has_variance_forest = function() {
      bart_samples_has_variance_forest_cpp(self$samples_ptr)
    },

    #' @description Global error variance samples (length `num_samples`, or empty).
    global_var_samples = function() {
      bart_samples_global_var_samples_cpp(self$samples_ptr)
    },

    #' @description Leaf scale samples (length `num_samples`, or empty).
    leaf_scale_samples = function() {
      bart_samples_leaf_scale_samples_cpp(self$samples_ptr)
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

    #' @description Borrowed (non-owning) external pointer to the mean forest container, for
    #' read-through prediction. Must not outlive this object.
    mean_forest_ptr = function() bart_samples_mean_forest_ptr_cpp(self$samples_ptr),

    #' @description Borrowed (non-owning) external pointer to the variance forest container.
    variance_forest_ptr = function() bart_samples_variance_forest_ptr_cpp(self$samples_ptr),

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
#' Container holding a sampled BCF model's forests and parameter traces as a single C++ object
#' (BCF analog of `BARTSamples`).
#' @noRd
BCFSamples <- R6::R6Class(
  classname = "BCFSamples",
  cloneable = FALSE,
  public = list(
    #' @field samples_ptr External pointer to a C++ BCFSamples object
    samples_ptr = NULL,

    #' @description
    #' Build a BCFSamples object by deep-copying existing forest containers and parameter arrays.
    #' @param mu_forest `ForestSamples` for the prognostic forest (required)
    #' @param tau_forest `ForestSamples` for the treatment forest (required)
    #' @param variance_forest `ForestSamples` for the variance forest (or NULL)
    #' @param global_var_samples Numeric vector of global error variance samples (or NULL)
    #' @param leaf_scale_mu_samples Numeric vector of prognostic leaf scale samples (or NULL)
    #' @param leaf_scale_tau_samples Numeric vector of treatment leaf scale samples (or NULL)
    #' @param tau_0_samples Numeric vector of treatment intercept samples (or NULL)
    #' @param b0_samples Numeric vector of adaptive-coding b0 samples (or NULL)
    #' @param b1_samples Numeric vector of adaptive-coding b1 samples (or NULL)
    #' @param y_bar Outcome mean used for standardization
    #' @param y_std Outcome standard deviation used for standardization
    #' @param num_samples Number of retained posterior samples
    #' @param treatment_dim Treatment dimension
    initialize = function(
      mu_forest,
      tau_forest,
      variance_forest = NULL,
      global_var_samples = NULL,
      leaf_scale_mu_samples = NULL,
      leaf_scale_tau_samples = NULL,
      tau_0_samples = NULL,
      b0_samples = NULL,
      b1_samples = NULL,
      y_bar = 0.0,
      y_std = 1.0,
      num_samples = 0L,
      treatment_dim = 1L
    ) {
      variance_ptr <- if (!is.null(variance_forest)) {
        variance_forest$forest_container_ptr
      } else {
        NULL
      }
      self$samples_ptr <- bcf_samples_from_components_cpp(
        mu_forest$forest_container_ptr,
        tau_forest$forest_container_ptr,
        variance_ptr,
        global_var_samples,
        leaf_scale_mu_samples,
        leaf_scale_tau_samples,
        tau_0_samples,
        b0_samples,
        b1_samples,
        y_bar,
        y_std,
        as.integer(num_samples),
        as.integer(treatment_dim)
      )
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
    has_tau_forest = function() bcf_samples_has_tau_forest_cpp(self$samples_ptr),

    #' @description Whether a variance forest is present.
    has_variance_forest = function() {
      bcf_samples_has_variance_forest_cpp(self$samples_ptr)
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

    #' @description Treatment intercept (tau_0) samples (flat).
    tau_0_samples = function() bcf_samples_tau_0_samples_cpp(self$samples_ptr),

    #' @description Adaptive-coding b0 samples.
    b0_samples = function() bcf_samples_b0_samples_cpp(self$samples_ptr),

    #' @description Adaptive-coding b1 samples.
    b1_samples = function() bcf_samples_b1_samples_cpp(self$samples_ptr),

    #' @description Materialize a deep copy of the prognostic forest as a `ForestSamples`.
    materialize_mu_forest = function() {
      if (!self$has_mu_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- bcf_samples_materialize_mu_forest_cpp(self$samples_ptr)
      fc
    },

    #' @description Materialize a deep copy of the treatment forest as a `ForestSamples`.
    materialize_tau_forest = function() {
      if (!self$has_tau_forest()) {
        return(NULL)
      }
      fc <- ForestSamples$new(0, 1, FALSE, FALSE)
      fc$forest_container_ptr <- bcf_samples_materialize_tau_forest_cpp(self$samples_ptr)
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

    #' @description Borrowed (non-owning) external pointer to the prognostic forest container.
    mu_forest_ptr = function() bcf_samples_mu_forest_ptr_cpp(self$samples_ptr),

    #' @description Borrowed (non-owning) external pointer to the treatment forest container.
    tau_forest_ptr = function() bcf_samples_tau_forest_ptr_cpp(self$samples_ptr),

    #' @description Borrowed (non-owning) external pointer to the variance forest container.
    variance_forest_ptr = function() bcf_samples_variance_forest_ptr_cpp(self$samples_ptr),

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
