#' Class that wraps the "persistent" aspects of a C++ random effects model
#' (draws of the parameters and a map from the original label indices to the 
#' 0-indexed label numbers used to place group samples in memory (i.e. the 
#' first label is stored in column 0 of the sample matrix, the second label 
#' is store in column 1 of the sample matrix, etc...))
#'
#' @description
#' Coordinates various C++ random effects classes and persists those 
#' needed for prediction / serialization

RandomEffectSamples <- R6::R6Class(
    classname = "RandomEffectSamples",
    cloneable = FALSE,
    public = list(
        
        #' @field rfx_container_ptr External pointer to a C++ StochTree::RandomEffectsContainer class
        rfx_container_ptr = NULL,
        
        #' @field label_mapper_ptr External pointer to a C++ StochTree::LabelMapper class
        label_mapper_ptr = NULL,
        
        #' @field training_group_ids Unique vector of group IDs that were in the training dataset
        training_group_ids = NULL,
        
        #' @description
        #' Create a new RandomEffectSamples object.
        #' @return A new `RandomEffectSamples` object.
        initialize = function() {}, 
        
        #' @description
        #' Construct RandomEffectSamples object from other "in-session" R objects
        #' @param num_components Number of "components" or bases defining the random effects regression
        #' @param num_groups Number of random effects groups
        #' @param random_effects_tracker Object of type `RandomEffectsTracker`
        #' @return None
        load_in_session = function(num_components, num_groups, random_effects_tracker) {
            # Initialize
            self$rfx_container_ptr <- rfx_container_cpp(num_components, num_groups)
            self$label_mapper_ptr <- rfx_label_mapper_cpp(random_effects_tracker$rfx_tracker_ptr)
            self$training_group_ids <- rfx_tracker_get_unique_group_ids_cpp(random_effects_tracker$rfx_tracker_ptr)
        }, 
        
        #' @description
        #' Construct RandomEffectSamples object from a json object
        #' @param json_object Object of class `CppJson`
        #' @param json_rfx_container_label Label referring to a particular rfx sample container (i.e. "random_effect_container_0") in the overall json hierarchy
        #' @param json_rfx_mapper_label Label referring to a particular rfx label mapper (i.e. "random_effect_label_mapper_0") in the overall json hierarchy
        #' @param json_rfx_groupids_label Label referring to a particular set of rfx group IDs (i.e. "random_effect_groupids_0") in the overall json hierarchy
        #' @return A new `RandomEffectSamples` object.
        load_from_json = function(json_object, json_rfx_container_label, json_rfx_mapper_label, json_rfx_groupids_label) {
            self$rfx_container_ptr <- rfx_container_from_json_cpp(json_object$json_ptr, json_rfx_container_label)
            self$label_mapper_ptr <- rfx_label_mapper_from_json_cpp(json_object$json_ptr, json_rfx_mapper_label)
            self$training_group_ids <- rfx_group_ids_from_json_cpp(json_object$json_ptr, json_rfx_groupids_label)
        }, 
        
        #' @description
        #' Append random effect draws to `RandomEffectSamples` object from a json object
        #' @param json_object Object of class `CppJson`
        #' @param json_rfx_container_label Label referring to a particular rfx sample container (i.e. "random_effect_container_0") in the overall json hierarchy
        #' @param json_rfx_mapper_label Label referring to a particular rfx label mapper (i.e. "random_effect_label_mapper_0") in the overall json hierarchy
        #' @param json_rfx_groupids_label Label referring to a particular set of rfx group IDs (i.e. "random_effect_groupids_0") in the overall json hierarchy
        #' @return None
        append_from_json = function(json_object, json_rfx_container_label, json_rfx_mapper_label, json_rfx_groupids_label) {
            rfx_container_append_from_json_cpp(self$rfx_container_ptr, json_object$json_ptr, json_rfx_container_label)
        }, 
        
        #' @description
        #' Construct RandomEffectSamples object from a json object
        #' @param json_string JSON string which parses into object of class `CppJson`
        #' @param json_rfx_container_label Label referring to a particular rfx sample container (i.e. "random_effect_container_0") in the overall json hierarchy
        #' @param json_rfx_mapper_label Label referring to a particular rfx label mapper (i.e. "random_effect_label_mapper_0") in the overall json hierarchy
        #' @param json_rfx_groupids_label Label referring to a particular set of rfx group IDs (i.e. "random_effect_groupids_0") in the overall json hierarchy
        #' @return A new `RandomEffectSamples` object.
        load_from_json_string = function(json_string, json_rfx_container_label, json_rfx_mapper_label, json_rfx_groupids_label) {
            self$rfx_container_ptr <- rfx_container_from_json_string_cpp(json_string, json_rfx_container_label)
            self$label_mapper_ptr <- rfx_label_mapper_from_json_string_cpp(json_string, json_rfx_mapper_label)
            self$training_group_ids <- rfx_group_ids_from_json_string_cpp(json_string, json_rfx_groupids_label)
        }, 
        
        #' @description
        #' Append random effect draws to `RandomEffectSamples` object from a json object
        #' @param json_string JSON string which parses into object of class `CppJson`
        #' @param json_rfx_container_label Label referring to a particular rfx sample container (i.e. "random_effect_container_0") in the overall json hierarchy
        #' @param json_rfx_mapper_label Label referring to a particular rfx label mapper (i.e. "random_effect_label_mapper_0") in the overall json hierarchy
        #' @param json_rfx_groupids_label Label referring to a particular set of rfx group IDs (i.e. "random_effect_groupids_0") in the overall json hierarchy
        #' @return None
        append_from_json_string = function(json_string, json_rfx_container_label, json_rfx_mapper_label, json_rfx_groupids_label) {
            # Append RFX objects
            rfx_container_append_from_json_string_cpp(self$rfx_container_ptr, json_string, json_rfx_container_label)
        }, 
        
        #' @description
        #' Predict random effects for each observation implied by `rfx_group_ids` and `rfx_basis`. 
        #' If a random effects model is "intercept-only" the `rfx_basis` will be a vector of ones of size `length(rfx_group_ids)`.
        #' @param rfx_group_ids Indices of random effects groups in a prediction set
        #' @param rfx_basis (Optional ) Basis used for random effects prediction
        #' @return Matrix with as many rows as observations provided and as many columns as samples drawn of the model.
        predict = function(rfx_group_ids, rfx_basis = NULL) {
            num_obs = length(rfx_group_ids)
            if (is.null(rfx_basis)) rfx_basis <- matrix(rep(1,num_obs), ncol = 1)
            num_samples = rfx_container_num_samples_cpp(self$rfx_container_ptr)
            num_components = rfx_container_num_components_cpp(self$rfx_container_ptr)
            num_groups = rfx_container_num_groups_cpp(self$rfx_container_ptr)
            rfx_group_ids_int <- as.integer(rfx_group_ids)
            stopifnot(sum(abs(rfx_group_ids_int-rfx_group_ids)) < 1e-6)
            stopifnot(sum(!(rfx_group_ids %in% self$training_group_ids)) == 0)
            stopifnot(ncol(rfx_basis) == num_components)
            rfx_dataset <- createRandomEffectsDataset(rfx_group_ids_int, rfx_basis)
            output <- rfx_container_predict_cpp(self$rfx_container_ptr, rfx_dataset$data_ptr, self$label_mapper_ptr)
            dim(output) <- c(num_obs, num_samples)
            return(output)
        }, 
        
        #' @description
        #' Extract the random effects parameters sampled. With the "redundant parameterization" 
        #' of Gelman et al (2008), this includes four parameters: alpha (the "working parameter" 
        #' shared across every group), xi (the "group parameter" sampled separately for each group), 
        #' beta (the product of alpha and xi, which corresponds to the overall group-level random effects), 
        #' and sigma (group-independent prior variance for each component of xi).
        #' @return List of arrays. The alpha array has dimension (`num_components`, `num_samples`) and is simply a vector if `num_components = 1`.
        #' The xi and beta arrays have dimension (`num_components`, `num_groups`, `num_samples`) and is simply a matrix if `num_components = 1`.
        #' The sigma array has dimension (`num_components`, `num_samples`) and is simply a vector if `num_components = 1`.
        extract_parameter_samples = function() {
            num_samples = rfx_container_num_samples_cpp(self$rfx_container_ptr)
            num_components = rfx_container_num_components_cpp(self$rfx_container_ptr)
            num_groups = rfx_container_num_groups_cpp(self$rfx_container_ptr)
            beta_samples <- rfx_container_get_beta_cpp(self$rfx_container_ptr)
            xi_samples <- rfx_container_get_xi_cpp(self$rfx_container_ptr)
            alpha_samples <- rfx_container_get_alpha_cpp(self$rfx_container_ptr)
            sigma_samples <- rfx_container_get_sigma_cpp(self$rfx_container_ptr)
            if (num_components == 1) {
                dim(beta_samples) <- c(num_groups, num_samples)
                dim(xi_samples) <- c(num_groups, num_samples)
            } else if (num_components > 1) {
                dim(beta_samples) <- c(num_components, num_groups, num_samples)
                dim(xi_samples) <- c(num_components, num_groups, num_samples)
                dim(alpha_samples) <- c(num_components, num_samples)
                dim(sigma_samples) <- c(num_components, num_samples)
            } else stop("Invalid random effects sample container, num_components is less than 1")
            
            output = list(
                "beta_samples" = beta_samples, 
                "xi_samples" = xi_samples, 
                "alpha_samples" = alpha_samples, 
                "sigma_samples" = sigma_samples
            )
            return(output)
        }, 
        
        #' @description
        #' Modify the `RandomEffectsSamples` object by removing the parameter samples index by `sample_num`.
        #' @param sample_num Index of the RFX sample to be removed
        delete_sample = function(sample_num) {
            rfx_container_delete_sample_cpp(self$rfx_container_ptr, sample_num)
        }, 
        
        #' @description
        #' Convert the mapping of group IDs to random effect components indices from C++ to R native format
        #' @return List mapping group ID to random effect components.
        extract_label_mapping = function() {
            keys_and_vals <- rfx_label_mapper_to_list_cpp(self$label_mapper_ptr)
            result <- as.list(keys_and_vals[[2]] + 1)
            setNames(result, keys_and_vals[[1]])
            return(result)
        }
    )
)

#' Class that defines a "tracker" for random effects models, most notably  
#' storing the data indices available in each group for quicker posterior 
#' computation and sampling of random effects terms.
#'
#' @description
#' Stores a mapping from every observation to its group index, a mapping 
#' from group indices to the training sample observations available in that 
#' group, and predictions for each observation.

RandomEffectsTracker <- R6::R6Class(
    classname = "RandomEffectsTracker",
    cloneable = FALSE,
    public = list(
        
        #' @field rfx_tracker_ptr External pointer to a C++ StochTree::RandomEffectsTracker class
        rfx_tracker_ptr = NULL,
        
        #' @description
        #' Create a new RandomEffectsTracker object.
        #' @param rfx_group_indices Integer indices indicating groups used to define random effects
        #' @return A new `RandomEffectsTracker` object.
        initialize = function(rfx_group_indices) {
            # Initialize
            self$rfx_tracker_ptr <- rfx_tracker_cpp(rfx_group_indices)
        }
    )
)

#' The core "model" class for sampling random effects.
#'
#' @description
#' Stores current model state, prior parameters, and procedures for 
#' sampling from the conditional posterior of each parameter.

RandomEffectsModel <- R6::R6Class(
    classname = "RandomEffectsModel",
    cloneable = FALSE,
    public = list(
        
        #' @field rfx_model_ptr External pointer to a C++ StochTree::RandomEffectsModel class
        rfx_model_ptr = NULL,
        
        #' @field num_groups Number of groups in the random effects model
        num_groups = NULL,
        
        #' @field num_components Number of components (i.e. dimension of basis) in the random effects model
        num_components = NULL,
        
        #' @description
        #' Create a new RandomEffectsModel object.
        #' @param num_components Number of "components" or bases defining the random effects regression
        #' @param num_groups Number of random effects groups
        #' @return A new `RandomEffectsModel` object.
        initialize = function(num_components, num_groups) {
            # Initialize
            self$rfx_model_ptr <- rfx_model_cpp(num_components, num_groups)
            self$num_components <- num_components
            self$num_groups <- num_groups
        },
        
        #' @description
        #' Sample from random effects model.
        #' @param rfx_dataset Object of type `RandomEffectsDataset`
        #' @param residual Object of type `Outcome`
        #' @param rfx_tracker Object of type `RandomEffectsTracker`
        #' @param rfx_samples Object of type `RandomEffectSamples`
        #' @param keep_sample Whether sample should be retained in `rfx_samples`. If `FALSE`, the state of `rfx_tracker` will be updated, but the parameter values will not be added to the sample container. Samples are commonly discarded due to burn-in or thinning.
        #' @param global_variance Scalar global variance parameter
        #' @param rng Object of type `CppRNG`
        #' @return None
        sample_random_effect = function(rfx_dataset, residual, rfx_tracker, rfx_samples, keep_sample, global_variance, rng) {
            rfx_model_sample_random_effects_cpp(self$rfx_model_ptr, rfx_dataset$data_ptr, 
                                                residual$data_ptr, rfx_tracker$rfx_tracker_ptr, 
                                                rfx_samples$rfx_container_ptr, keep_sample, global_variance, rng$rng_ptr)
        },
        
        #' @description
        #' Predict from (a single sample of a) random effects model.
        #' @param rfx_dataset Object of type `RandomEffectsDataset`
        #' @param rfx_tracker Object of type `RandomEffectsTracker`
        #' @return Vector of predictions with size matching number of observations in rfx_dataset
        predict = function(rfx_dataset, rfx_tracker) {
            pred <- rfx_model_predict_cpp(self$rfx_model_ptr, rfx_dataset$data_ptr, rfx_tracker$rfx_tracker_ptr)
            return(pred)
        },
        
        #' @description
        #' Set value for the "working parameter." This is typically 
        #' used for initialization, but could also be used to interrupt 
        #' or override the sampler.
        #' @param value Parameter input
        #' @return None
        set_working_parameter = function(value) {
            stopifnot(is.double(value))
            stopifnot(!is.matrix(value))
            stopifnot(length(value) == self$num_components)
            rfx_model_set_working_parameter_cpp(self$rfx_model_ptr, value)
        },
        
        #' @description
        #' Set value for the "group parameters." This is typically 
        #' used for initialization, but could also be used to interrupt 
        #' or override the sampler.
        #' @param value Parameter input
        #' @return None
        set_group_parameters = function(value) {
            stopifnot(is.double(value))
            stopifnot(is.matrix(value))
            stopifnot(nrow(value) == self$num_components)
            stopifnot(ncol(value) == self$num_groups)
            rfx_model_set_group_parameters_cpp(self$rfx_model_ptr, value)
        },
        
        #' @description
        #' Set value for the working parameter covariance. This is typically 
        #' used for initialization, but could also be used to interrupt 
        #' or override the sampler.
        #' @param value Parameter input
        #' @return None
        set_working_parameter_cov = function(value) {
            stopifnot(is.double(value))
            stopifnot(is.matrix(value))
            stopifnot(nrow(value) == self$num_components)
            stopifnot(ncol(value) == self$num_components)
            rfx_model_set_working_parameter_covariance_cpp(self$rfx_model_ptr, value)
        },
        
        #' @description
        #' Set value for the group parameter covariance. This is typically 
        #' used for initialization, but could also be used to interrupt 
        #' or override the sampler.
        #' @param value Parameter input
        #' @return None
        set_group_parameter_cov = function(value) {
            stopifnot(is.double(value))
            stopifnot(is.matrix(value))
            stopifnot(nrow(value) == self$num_components)
            stopifnot(ncol(value) == self$num_components)
            rfx_model_set_group_parameter_covariance_cpp(self$rfx_model_ptr, value)
        }, 
        
        #' @description
        #' Set shape parameter for the group parameter variance prior.
        #' @param value Parameter input
        #' @return None
        set_variance_prior_shape = function(value) {
            stopifnot(is.double(value))
            stopifnot(!is.matrix(value))
            stopifnot(length(value) == 1)
            rfx_model_set_variance_prior_shape_cpp(self$rfx_model_ptr, value)
        },
        
        #' @description
        #' Set shape parameter for the group parameter variance prior.
        #' @param value Parameter input
        #' @return None
        set_variance_prior_scale = function(value) {
            stopifnot(is.double(value))
            stopifnot(!is.matrix(value))
            stopifnot(length(value) == 1)
            rfx_model_set_variance_prior_scale_cpp(self$rfx_model_ptr, value)
        }
    )
)

#' Create a `RandomEffectSamples` object
#'
#' @param num_components Number of "components" or bases defining the random effects regression
#' @param num_groups Number of random effects groups
#' @param random_effects_tracker Object of type `RandomEffectsTracker`
#' @return `RandomEffectSamples` object
#' @export
#' 
#' @examples
#' n <- 100
#' rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
#' rfx_basis <- matrix(rep(1.0, n), ncol=1)
#' num_groups <- length(unique(rfx_group_ids))
#' num_components <- ncol(rfx_basis)
#' rfx_tracker <- createRandomEffectsTracker(rfx_group_ids)
#' rfx_samples <- createRandomEffectSamples(num_components, num_groups, rfx_tracker)
createRandomEffectSamples <- function(num_components, num_groups, random_effects_tracker) {
    invisible(output <- RandomEffectSamples$new())
    output$load_in_session(num_components, num_groups, random_effects_tracker)
    return(output)
}

#' Create a `RandomEffectsTracker` object
#'
#' @param rfx_group_indices Integer indices indicating groups used to define random effects
#' @return `RandomEffectsTracker` object
#' @export
#' 
#' @examples
#' n <- 100
#' rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
#' rfx_basis <- matrix(rep(1.0, n), ncol=1)
#' num_groups <- length(unique(rfx_group_ids))
#' num_components <- ncol(rfx_basis)
#' rfx_tracker <- createRandomEffectsTracker(rfx_group_ids)
createRandomEffectsTracker <- function(rfx_group_indices) {
    return(invisible((
        RandomEffectsTracker$new(rfx_group_indices)
    )))
}

#' Create a `RandomEffectsModel` object
#'
#' @param num_components Number of "components" or bases defining the random effects regression
#' @param num_groups Number of random effects groups
#' @return `RandomEffectsModel` object
#' @export
#' 
#' @examples
#' n <- 100
#' rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
#' rfx_basis <- matrix(rep(1.0, n), ncol=1)
#' num_groups <- length(unique(rfx_group_ids))
#' num_components <- ncol(rfx_basis)
#' rfx_model <- createRandomEffectsModel(num_components, num_groups)
createRandomEffectsModel <- function(num_components, num_groups) {
    return(invisible((
        RandomEffectsModel$new(num_components, num_groups)
    )))
}

#' Reset a `RandomEffectsModel` object based on the parameters indexed by `sample_num` in a `RandomEffectsSamples` object
#'
#' @param rfx_model Object of type `RandomEffectsModel`.
#' @param rfx_samples Object of type `RandomEffectSamples`.
#' @param sample_num Index of sample stored in `rfx_samples` from which to reset the state of a random effects model. Zero-indexed, so resetting based on the first sample would require setting `sample_num = 0`.
#' @param sigma_alpha_init Initial value of the "working parameter" scale parameter.
#' @return None
#' @export
#' 
#' @examples
#' n <- 100
#' p <- 10
#' rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
#' rfx_basis <- matrix(rep(1.0, n), ncol=1)
#' rfx_dataset <- createRandomEffectsDataset(rfx_group_ids, rfx_basis)
#' y <- (-2*(rfx_group_ids==1)+2*(rfx_group_ids==2)) + rnorm(n)
#' y_std <- (y-mean(y))/sd(y)
#' outcome <- createOutcome(y_std)
#' rng <- createCppRNG(1234)
#' num_groups <- length(unique(rfx_group_ids))
#' num_components <- ncol(rfx_basis)
#' rfx_model <- createRandomEffectsModel(num_components, num_groups)
#' rfx_tracker <- createRandomEffectsTracker(rfx_group_ids)
#' rfx_samples <- createRandomEffectSamples(num_components, num_groups, rfx_tracker)
#' alpha_init <- rep(1,num_components)
#' xi_init <- matrix(rep(alpha_init, num_groups),num_components,num_groups)
#' sigma_alpha_init <- diag(1,num_components,num_components)
#' sigma_xi_init <- diag(1,num_components,num_components)
#' sigma_xi_shape <- 1
#' sigma_xi_scale <- 1
#' rfx_model$set_working_parameter(alpha_init)
#' rfx_model$set_group_parameters(xi_init)
#' rfx_model$set_working_parameter_cov(sigma_alpha_init)
#' rfx_model$set_group_parameter_cov(sigma_xi_init)
#' rfx_model$set_variance_prior_shape(sigma_xi_shape)
#' rfx_model$set_variance_prior_scale(sigma_xi_scale)
#' for (i in 1:3) {
#'     rfx_model$sample_random_effect(rfx_dataset=rfx_dataset, residual=outcome, 
#'                                    rfx_tracker=rfx_tracker, rfx_samples=rfx_samples, 
#'                                    keep_sample=TRUE, global_variance=1.0, rng=rng)
#' }
#' resetRandomEffectsModel(rfx_model, rfx_samples, 0, 1.0)
resetRandomEffectsModel <- function(rfx_model, rfx_samples, sample_num, sigma_alpha_init) {
    if (!is.matrix(sigma_alpha_init)) {
        if (!is.double(sigma_alpha_init)) {
            stop("`sigma_alpha_init` must be a numeric scalar or matrix")
        }
        sigma_alpha_init <- as.matrix(sigma_alpha_init)
    }
    reset_rfx_model_cpp(rfx_model$rfx_model_ptr, rfx_samples$rfx_container_ptr, sample_num)
    rfx_model$set_working_parameter_cov(sigma_alpha_init)
}

#' Reset a `RandomEffectsTracker` object based on the parameters indexed by `sample_num` in a `RandomEffectsSamples` object
#'
#' @param rfx_tracker Object of type `RandomEffectsTracker`.
#' @param rfx_model Object of type `RandomEffectsModel`.
#' @param rfx_dataset Object of type `RandomEffectsDataset`.
#' @param residual Object of type `Outcome`.
#' @param rfx_samples Object of type `RandomEffectSamples`.
#' @return None
#' @export
#' 
#' @examples
#' n <- 100
#' p <- 10
#' rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
#' rfx_basis <- matrix(rep(1.0, n), ncol=1)
#' rfx_dataset <- createRandomEffectsDataset(rfx_group_ids, rfx_basis)
#' y <- (-2*(rfx_group_ids==1)+2*(rfx_group_ids==2)) + rnorm(n)
#' y_std <- (y-mean(y))/sd(y)
#' outcome <- createOutcome(y_std)
#' rng <- createCppRNG(1234)
#' num_groups <- length(unique(rfx_group_ids))
#' num_components <- ncol(rfx_basis)
#' rfx_model <- createRandomEffectsModel(num_components, num_groups)
#' rfx_tracker <- createRandomEffectsTracker(rfx_group_ids)
#' rfx_samples <- createRandomEffectSamples(num_components, num_groups, rfx_tracker)
#' alpha_init <- rep(1,num_components)
#' xi_init <- matrix(rep(alpha_init, num_groups),num_components,num_groups)
#' sigma_alpha_init <- diag(1,num_components,num_components)
#' sigma_xi_init <- diag(1,num_components,num_components)
#' sigma_xi_shape <- 1
#' sigma_xi_scale <- 1
#' rfx_model$set_working_parameter(alpha_init)
#' rfx_model$set_group_parameters(xi_init)
#' rfx_model$set_working_parameter_cov(sigma_alpha_init)
#' rfx_model$set_group_parameter_cov(sigma_xi_init)
#' rfx_model$set_variance_prior_shape(sigma_xi_shape)
#' rfx_model$set_variance_prior_scale(sigma_xi_scale)
#' for (i in 1:3) {
#'     rfx_model$sample_random_effect(rfx_dataset=rfx_dataset, residual=outcome, 
#'                                    rfx_tracker=rfx_tracker, rfx_samples=rfx_samples, 
#'                                    keep_sample=TRUE, global_variance=1.0, rng=rng)
#' }
#' resetRandomEffectsModel(rfx_model, rfx_samples, 0, 1.0)
#' resetRandomEffectsTracker(rfx_tracker, rfx_model, rfx_dataset, outcome, rfx_samples)
resetRandomEffectsTracker <- function(rfx_tracker, rfx_model, rfx_dataset, residual, rfx_samples) {
    reset_rfx_tracker_cpp(rfx_tracker$rfx_tracker_ptr, rfx_dataset$data_ptr, residual$data_ptr, rfx_model$rfx_model_ptr)
}

#' Reset a `RandomEffectsModel` object to its "default" state
#'
#' @param rfx_model Object of type `RandomEffectsModel`.
#' @param alpha_init Initial value of the "working parameter".
#' @param xi_init Initial value of the "group parameters".
#' @param sigma_alpha_init Initial value of the "working parameter" scale parameter.
#' @param sigma_xi_init Initial value of the "group parameters" scale parameter.
#' @param sigma_xi_shape Shape parameter for the inverse gamma variance model on the group parameters.
#' @param sigma_xi_scale Scale parameter for the inverse gamma variance model on the group parameters.
#' @return None
#' @export
#' 
#' @examples
#' n <- 100
#' p <- 10
#' rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
#' rfx_basis <- matrix(rep(1.0, n), ncol=1)
#' rfx_dataset <- createRandomEffectsDataset(rfx_group_ids, rfx_basis)
#' y <- (-2*(rfx_group_ids==1)+2*(rfx_group_ids==2)) + rnorm(n)
#' y_std <- (y-mean(y))/sd(y)
#' outcome <- createOutcome(y_std)
#' rng <- createCppRNG(1234)
#' num_groups <- length(unique(rfx_group_ids))
#' num_components <- ncol(rfx_basis)
#' rfx_model <- createRandomEffectsModel(num_components, num_groups)
#' rfx_tracker <- createRandomEffectsTracker(rfx_group_ids)
#' rfx_samples <- createRandomEffectSamples(num_components, num_groups, rfx_tracker)
#' alpha_init <- rep(1,num_components)
#' xi_init <- matrix(rep(alpha_init, num_groups),num_components,num_groups)
#' sigma_alpha_init <- diag(1,num_components,num_components)
#' sigma_xi_init <- diag(1,num_components,num_components)
#' sigma_xi_shape <- 1
#' sigma_xi_scale <- 1
#' rfx_model$set_working_parameter(alpha_init)
#' rfx_model$set_group_parameters(xi_init)
#' rfx_model$set_working_parameter_cov(sigma_alpha_init)
#' rfx_model$set_group_parameter_cov(sigma_xi_init)
#' rfx_model$set_variance_prior_shape(sigma_xi_shape)
#' rfx_model$set_variance_prior_scale(sigma_xi_scale)
#' for (i in 1:3) {
#'     rfx_model$sample_random_effect(rfx_dataset=rfx_dataset, residual=outcome, 
#'                                    rfx_tracker=rfx_tracker, rfx_samples=rfx_samples, 
#'                                    keep_sample=TRUE, global_variance=1.0, rng=rng)
#' }
#' rootResetRandomEffectsModel(rfx_model, alpha_init, xi_init, sigma_alpha_init,
#'                             sigma_xi_init, sigma_xi_shape, sigma_xi_scale)
rootResetRandomEffectsModel <- function(rfx_model, alpha_init, xi_init, sigma_alpha_init,
                                        sigma_xi_init, sigma_xi_shape, sigma_xi_scale) {
    rfx_model$set_working_parameter(alpha_init)
    rfx_model$set_group_parameters(xi_init)
    rfx_model$set_working_parameter_cov(sigma_alpha_init)
    rfx_model$set_group_parameter_cov(sigma_xi_init)
    rfx_model$set_variance_prior_shape(sigma_xi_shape)
    rfx_model$set_variance_prior_scale(sigma_xi_scale)
}

#' Reset a `RandomEffectsTracker` object to its "default" state
#'
#' @param rfx_tracker Object of type `RandomEffectsTracker`.
#' @param rfx_model Object of type `RandomEffectsModel`.
#' @param rfx_dataset Object of type `RandomEffectsDataset`.
#' @param residual Object of type `Outcome`.
#' @return None
#' @export
#' 
#' @examples
#' n <- 100
#' p <- 10
#' rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
#' rfx_basis <- matrix(rep(1.0, n), ncol=1)
#' rfx_dataset <- createRandomEffectsDataset(rfx_group_ids, rfx_basis)
#' y <- (-2*(rfx_group_ids==1)+2*(rfx_group_ids==2)) + rnorm(n)
#' y_std <- (y-mean(y))/sd(y)
#' outcome <- createOutcome(y_std)
#' rng <- createCppRNG(1234)
#' num_groups <- length(unique(rfx_group_ids))
#' num_components <- ncol(rfx_basis)
#' rfx_model <- createRandomEffectsModel(num_components, num_groups)
#' rfx_tracker <- createRandomEffectsTracker(rfx_group_ids)
#' rfx_samples <- createRandomEffectSamples(num_components, num_groups, rfx_tracker)
#' alpha_init <- rep(1,num_components)
#' xi_init <- matrix(rep(alpha_init, num_groups),num_components,num_groups)
#' sigma_alpha_init <- diag(1,num_components,num_components)
#' sigma_xi_init <- diag(1,num_components,num_components)
#' sigma_xi_shape <- 1
#' sigma_xi_scale <- 1
#' rfx_model$set_working_parameter(alpha_init)
#' rfx_model$set_group_parameters(xi_init)
#' rfx_model$set_working_parameter_cov(sigma_alpha_init)
#' rfx_model$set_group_parameter_cov(sigma_xi_init)
#' rfx_model$set_variance_prior_shape(sigma_xi_shape)
#' rfx_model$set_variance_prior_scale(sigma_xi_scale)
#' for (i in 1:3) {
#'     rfx_model$sample_random_effect(rfx_dataset=rfx_dataset, residual=outcome, 
#'                                    rfx_tracker=rfx_tracker, rfx_samples=rfx_samples, 
#'                                    keep_sample=TRUE, global_variance=1.0, rng=rng)
#' }
#' rootResetRandomEffectsModel(rfx_model, alpha_init, xi_init, sigma_alpha_init,
#'                             sigma_xi_init, sigma_xi_shape, sigma_xi_scale)
#' rootResetRandomEffectsTracker(rfx_tracker, rfx_model, rfx_dataset, outcome)
rootResetRandomEffectsTracker <- function(rfx_tracker, rfx_model, rfx_dataset, residual) {
    root_reset_rfx_tracker_cpp(rfx_tracker$rfx_tracker_ptr, rfx_dataset$data_ptr, residual$data_ptr, rfx_model$rfx_model_ptr)
}
