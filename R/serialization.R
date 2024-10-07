#' Class that stores draws from an random ensemble of decision trees
#'
#' @description
#' Wrapper around a C++ container of tree ensembles

CppJson <- R6::R6Class(
    classname = "CppJson",
    cloneable = FALSE,
    public = list(
        
        #' @field json_ptr External pointer to a C++ nlohmann::json object
        json_ptr = NULL,
        
        #' @field num_forests Number of forests in the nlohmann::json object
        num_forests = NULL,
        
        #' @field forest_labels Names of forest objects in the overall nlohmann::json object
        forest_labels = NULL,
        
        #' @field num_rfx Number of random effects terms in the nlohman::json object
        num_rfx = NULL,
        
        #' @field rfx_container_labels Names of rfx container objects in the overall nlohmann::json object
        rfx_container_labels = NULL,
        
        #' @field rfx_mapper_labels Names of rfx label mapper objects in the overall nlohmann::json object
        rfx_mapper_labels = NULL,
        
        #' @field rfx_groupid_labels Names of rfx group id objects in the overall nlohmann::json object
        rfx_groupid_labels = NULL,
        
        #' @description
        #' Create a new CppJson object.
        #' @return A new `CppJson` object.
        initialize = function() {
            self$json_ptr <- init_json_cpp()
            self$num_forests <- 0
            self$forest_labels <- c()
            self$num_rfx <- 0
            self$rfx_container_labels <- c()
            self$rfx_mapper_labels <- c()
            self$rfx_groupid_labels <- c()
        }, 
        
        #' @description
        #' Convert a forest container to json and add to the current `CppJson` object
        #' @param forest_samples `ForestSamples` R class
        #' @return NULL
        add_forest = function(forest_samples) {
            forest_label <- json_add_forest_cpp(self$json_ptr, forest_samples$forest_container_ptr)
            self$num_forests <- self$num_forests + 1
            self$forest_labels <- c(self$forest_labels, forest_label)
        }, 
        
        #' @description
        #' Convert a random effects container to json and add to the current `CppJson` object
        #' @param rfx_samples `RandomEffectSamples` R class
        #' @return NULL
        add_random_effects = function(rfx_samples) {
            rfx_container_label <- json_add_rfx_container_cpp(self$json_ptr, rfx_samples$rfx_container_ptr)
            self$rfx_container_labels <- c(self$rfx_container_labels, rfx_container_label)
            rfx_mapper_label <- json_add_rfx_label_mapper_cpp(self$json_ptr, rfx_samples$label_mapper_ptr)
            self$rfx_mapper_labels <- c(self$rfx_mapper_labels, rfx_mapper_label)
            rfx_groupid_label <- json_add_rfx_groupids_cpp(self$json_ptr, rfx_samples$training_group_ids)
            self$rfx_groupid_labels <- c(self$rfx_groupid_labels, rfx_groupid_label)
            json_increment_rfx_count_cpp(self$json_ptr)
            self$num_rfx <- self$num_rfx + 1
        }, 
        
        #' @description
        #' Add a scalar to the json object under the name "field_name" (with optional subfolder "subfolder_name")
        #' @param field_name The name of the field to be added to json
        #' @param field_value Numeric value of the field to be added to json
        #' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which to place the value
        #' @return NULL
        add_scalar = function(field_name, field_value, subfolder_name = NULL) {
            if (is.null(subfolder_name)) {
                json_add_double_cpp(self$json_ptr, field_name, field_value)
            } else {
                json_add_double_subfolder_cpp(self$json_ptr, subfolder_name, field_name, field_value)
            }
        }, 
        
        #' @description
        #' Add a boolean value to the json object under the name "field_name" (with optional subfolder "subfolder_name")
        #' @param field_name The name of the field to be added to json
        #' @param field_value Numeric value of the field to be added to json
        #' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which to place the value
        #' @return NULL
        add_boolean = function(field_name, field_value, subfolder_name = NULL) {
            if (is.null(subfolder_name)) {
                json_add_bool_cpp(self$json_ptr, field_name, field_value)
            } else {
                json_add_bool_subfolder_cpp(self$json_ptr, subfolder_name, field_name, field_value)
            }
        }, 
        
        #' @description
        #' Add a string value to the json object under the name "field_name" (with optional subfolder "subfolder_name")
        #' @param field_name The name of the field to be added to json
        #' @param field_value Numeric value of the field to be added to json
        #' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which to place the value
        #' @return NULL
        add_string = function(field_name, field_value, subfolder_name = NULL) {
            if (is.null(subfolder_name)) {
                json_add_string_cpp(self$json_ptr, field_name, field_value)
            } else {
                json_add_string_subfolder_cpp(self$json_ptr, subfolder_name, field_name, field_value)
            }
        }, 
        
        #' @description
        #' Add an array to the json object under the name "field_name" (with optional subfolder "subfolder_name")
        #' @param field_name The name of the field to be added to json
        #' @param field_vector Vector to be stored in json
        #' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which to place the value
        #' @return NULL
        add_vector = function(field_name, field_vector, subfolder_name = NULL) {
            field_vector <- as.numeric(field_vector)
            if (is.null(subfolder_name)) {
                json_add_vector_cpp(self$json_ptr, field_name, field_vector)
            } else {
                json_add_vector_subfolder_cpp(self$json_ptr, subfolder_name, field_name, field_vector)
            }
        }, 
        
        #' @description
        #' Add an array to the json object under the name "field_name" (with optional subfolder "subfolder_name")
        #' @param field_name The name of the field to be added to json
        #' @param field_vector Character vector to be stored in json
        #' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which to place the value
        #' @return NULL
        add_string_vector = function(field_name, field_vector, subfolder_name = NULL) {
            if (is.null(subfolder_name)) {
                json_add_string_vector_cpp(self$json_ptr, field_name, field_vector)
            } else {
                json_add_string_vector_subfolder_cpp(self$json_ptr, subfolder_name, field_name, field_vector)
            }
        }, 
        
        #' @description
        #' Add a list of vectors (as an object map of arrays) to the json object under the name "field_name"
        #' @param field_name The name of the field to be added to json
        #' @param field_list List to be stored in json
        #' @return NULL
        add_list = function(field_name, field_list) {
            stopifnot(sum(!sapply(field_list, is.vector))==0)
            list_names <- names(field_list)
            for (i in 1:length(field_list)) {
                vec_name <- list_names[i]
                vec <- field_list[[i]]
                json_add_vector_subfolder_cpp(self$json_ptr, field_name, vec_name, vec)
            }
        }, 
        
        #' @description
        #' Add a list of vectors (as an object map of arrays) to the json object under the name "field_name"
        #' @param field_name The name of the field to be added to json
        #' @param field_list List to be stored in json
        #' @return NULL
        add_string_list = function(field_name, field_list) {
            stopifnot(sum(!sapply(field_list, is.vector))==0)
            list_names <- names(field_list)
            for (i in 1:length(field_list)) {
                vec_name <- list_names[i]
                vec <- field_list[[i]]
                json_add_string_vector_subfolder_cpp(self$json_ptr, field_name, vec_name, vec)
            }
        }, 
        
        #' @description
        #' Retrieve a scalar value from the json object under the name "field_name" (with optional subfolder "subfolder_name")
        #' @param field_name The name of the field to be accessed from json
        #' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which the field is stored
        #' @return NULL
        get_scalar = function(field_name, subfolder_name = NULL) {
            if (is.null(subfolder_name)) {
                stopifnot(json_contains_field_cpp(self$json_ptr, field_name))
                result <- json_extract_double_cpp(self$json_ptr, field_name)
            } else {
                stopifnot(json_contains_field_subfolder_cpp(self$json_ptr, subfolder_name, field_name))
                result <- json_extract_double_subfolder_cpp(self$json_ptr, subfolder_name, field_name)
            }
            return(result)
        }, 
        
        #' @description
        #' Retrieve a boolean value from the json object under the name "field_name" (with optional subfolder "subfolder_name")
        #' @param field_name The name of the field to be accessed from json
        #' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which the field is stored
        #' @return NULL
        get_boolean = function(field_name, subfolder_name = NULL) {
            if (is.null(subfolder_name)) {
                stopifnot(json_contains_field_cpp(self$json_ptr, field_name))
                result <- json_extract_bool_cpp(self$json_ptr, field_name)
            } else {
                stopifnot(json_contains_field_subfolder_cpp(self$json_ptr, subfolder_name, field_name))
                result <- json_extract_bool_subfolder_cpp(self$json_ptr, subfolder_name, field_name)
            }
            return(result)
        }, 
        
        #' @description
        #' Retrieve a string value from the json object under the name "field_name" (with optional subfolder "subfolder_name")
        #' @param field_name The name of the field to be accessed from json
        #' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which the field is stored
        #' @return NULL
        get_string = function(field_name, subfolder_name = NULL) {
            if (is.null(subfolder_name)) {
                stopifnot(json_contains_field_cpp(self$json_ptr, field_name))
                result <- json_extract_string_cpp(self$json_ptr, field_name)
            } else {
                stopifnot(json_contains_field_subfolder_cpp(self$json_ptr, subfolder_name, field_name))
                result <- json_extract_string_subfolder_cpp(self$json_ptr, subfolder_name, field_name)
            }
            return(result)
        }, 
        
        #' @description
        #' Retrieve a vector from the json object under the name "field_name" (with optional subfolder "subfolder_name")
        #' @param field_name The name of the field to be accessed from json
        #' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which the field is stored
        #' @return NULL
        get_vector = function(field_name, subfolder_name = NULL) {
            if (is.null(subfolder_name)) {
                stopifnot(json_contains_field_cpp(self$json_ptr, field_name))
                result <- json_extract_vector_cpp(self$json_ptr, field_name)
            } else {
                stopifnot(json_contains_field_subfolder_cpp(self$json_ptr, subfolder_name, field_name))
                result <- json_extract_vector_subfolder_cpp(self$json_ptr, subfolder_name, field_name)
            }
            return(result)
        }, 
        
        #' @description
        #' Retrieve a character vector from the json object under the name "field_name" (with optional subfolder "subfolder_name")
        #' @param field_name The name of the field to be accessed from json
        #' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which the field is stored
        #' @return NULL
        get_string_vector = function(field_name, subfolder_name = NULL) {
            if (is.null(subfolder_name)) {
                stopifnot(json_contains_field_cpp(self$json_ptr, field_name))
                result <- json_extract_string_vector_cpp(self$json_ptr, field_name)
            } else {
                stopifnot(json_contains_field_subfolder_cpp(self$json_ptr, subfolder_name, field_name))
                result <- json_extract_string_vector_subfolder_cpp(self$json_ptr, subfolder_name, field_name)
            }
            return(result)
        }, 
        
        #' @description
        #' Reconstruct a list of numeric vectors from the json object stored under "field_name"
        #' @param field_name The name of the field to be added to json
        #' @param key_names Vector of names of list elements (each of which is a vector)
        #' @return NULL
        get_numeric_list = function(field_name, key_names) {
            output <- list()
            for (i in 1:length(key_names)) {
                vec_name <- key_names[i]
                output[[vec_name]] <- json_extract_vector_subfolder_cpp(self$json_ptr, field_name, vec_name)
            }
            return(output)
        }, 
        
        #' @description
        #' Reconstruct a list of string vectors from the json object stored under "field_name"
        #' @param field_name The name of the field to be added to json
        #' @param key_names Vector of names of list elements (each of which is a vector)
        #' @return NULL
        get_string_list = function(field_name, key_names) {
            output <- list()
            for (i in 1:length(key_names)) {
                vec_name <- key_names[i]
                output[[vec_name]] <- json_extract_string_vector_subfolder_cpp(self$json_ptr, field_name, vec_name)
            }
            return(output)
        }, 
        
        #' @description
        #' Convert a JSON object to in-memory string
        #' @return JSON string
        return_json_string = function() {
            return(get_json_string_cpp(self$json_ptr))
        }, 
        
        #' @description
        #' Save a json object to file
        #' @param filename String of filepath, must end in ".json"
        #' @return NULL
        save_file = function(filename) {
            json_save_file_cpp(self$json_ptr, filename)
        }, 
        
        #' @description
        #' Load a json object from file
        #' @param filename String of filepath, must end in ".json"
        #' @return NULL
        load_from_file = function(filename) {
            json_load_file_cpp(self$json_ptr, filename)
        }, 
        
        #' @description
        #' Load a json object from string
        #' @param json_string JSON string dump
        #' @return NULL
        load_from_string = function(json_string) {
            json_load_string_cpp(self$json_ptr, json_string)
        }
    )
)

#' Load a container of forest samples from json
#'
#' @param json_object Object of class `CppJson`
#' @param json_forest_label Label referring to a particular forest (i.e. "forest_0") in the overall json hierarchy
#'
#' @return `ForestSamples` object
#' @export
loadForestContainerJson <- function(json_object, json_forest_label) {
    invisible(output <- ForestSamples$new(0,1,T))
    output$load_from_json(json_object, json_forest_label)
    return(output)
}

#' Combine multiple JSON model objects containing forests (with the same hierarchy / schema) into a single forest_container
#'
#' @param json_object_list List of objects of class `CppJson`
#' @param json_forest_label Label referring to a particular forest (i.e. "forest_0") in the overall json hierarchy (must exist in every json object in the list)
#'
#' @return `ForestSamples` object
#' @export
loadForestContainerCombinedJson <- function(json_object_list, json_forest_label) {
    invisible(output <- ForestSamples$new(0,1,T))
    for (i in 1:length(json_object_list)) {
        json_object <- json_object_list[[i]]
        if (i == 1) {
            output$load_from_json(json_object, json_forest_label)
        } else {
            output$append_from_json(json_object, json_forest_label)
        }
    }
    return(output)
}

#' Combine multiple JSON strings representing model objects containing forests (with the same hierarchy / schema) into a single forest_container
#'
#' @param json_string_list List of strings that parse into objects of type `CppJson`
#' @param json_forest_label Label referring to a particular forest (i.e. "forest_0") in the overall json hierarchy (must exist in every json object in the list)
#'
#' @return `ForestSamples` object
#' @export
loadForestContainerCombinedJsonString <- function(json_string_list, json_forest_label) {
    invisible(output <- ForestSamples$new(0,1,T))
    for (i in 1:length(json_string_list)) {
        json_string <- json_string_list[[i]]
        if (i == 1) {
            output$load_from_json_string(json_string, json_forest_label)
        } else {
            output$append_from_json_string(json_string, json_forest_label)
        }
    }
    return(output)
}

#' Load a container of random effect samples from json
#'
#' @param json_object Object of class `CppJson`
#' @param json_rfx_num Integer index indicating the position of the random effects term to be unpacked
#'
#' @return `RandomEffectSamples` object
#' @export
loadRandomEffectSamplesJson <- function(json_object, json_rfx_num) {
    json_rfx_container_label <- paste0("random_effect_container_", json_rfx_num)
    json_rfx_mapper_label <- paste0("random_effect_label_mapper_", json_rfx_num)
    json_rfx_groupids_label <- paste0("random_effect_groupids_", json_rfx_num)
    invisible(output <- RandomEffectSamples$new())
    output$load_from_json(json_object, json_rfx_container_label, json_rfx_mapper_label, json_rfx_groupids_label)
    return(output)
}

#' Combine multiple JSON model objects containing random effects (with the same hierarchy / schema) into a single container
#'
#' @param json_object_list List of objects of class `CppJson`
#' @param json_rfx_num Integer index indicating the position of the random effects term to be unpacked
#'
#' @return `RandomEffectSamples` object
#' @export
loadRandomEffectSamplesCombinedJson <- function(json_object_list, json_rfx_num) {
    json_rfx_container_label <- paste0("random_effect_container_", json_rfx_num)
    json_rfx_mapper_label <- paste0("random_effect_label_mapper_", json_rfx_num)
    json_rfx_groupids_label <- paste0("random_effect_groupids_", json_rfx_num)
    invisible(output <- RandomEffectSamples$new())
    for (i in 1:length(json_object_list)) {
        json_object <- json_object_list[[i]]
        if (i == 1) {
            output$load_from_json(json_object, json_rfx_container_label, json_rfx_mapper_label, json_rfx_groupids_label)
        } else {
            output$append_from_json(json_object, json_rfx_container_label, json_rfx_mapper_label, json_rfx_groupids_label)
        }
    }
    return(output)
}

#' Combine multiple JSON strings representing model objects containing random effects (with the same hierarchy / schema) into a single container
#'
#' @param json_string_list List of objects of class `CppJson`
#' @param json_rfx_num Integer index indicating the position of the random effects term to be unpacked
#'
#' @return `RandomEffectSamples` object
#' @export
loadRandomEffectSamplesCombinedJsonString <- function(json_string_list, json_rfx_num) {
    json_rfx_container_label <- paste0("random_effect_container_", json_rfx_num)
    json_rfx_mapper_label <- paste0("random_effect_label_mapper_", json_rfx_num)
    json_rfx_groupids_label <- paste0("random_effect_groupids_", json_rfx_num)
    invisible(output <- RandomEffectSamples$new())
    for (i in 1:length(json_object_list)) {
        json_string <- json_string_list[[i]]
        if (i == 1) {
            output$load_from_json_string(json_string, json_rfx_container_label, json_rfx_mapper_label, json_rfx_groupids_label)
        } else {
            output$append_from_json_string(json_string, json_rfx_container_label, json_rfx_mapper_label, json_rfx_groupids_label)
        }
    }
    return(output)
}

#' Load a vector from json
#'
#' @param json_object Object of class `CppJson`
#' @param json_vector_label Label referring to a particular vector (i.e. "sigma2_samples") in the overall json hierarchy
#' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which vector sits
#'
#' @return R vector
#' @export
loadVectorJson <- function(json_object, json_vector_label, subfolder_name = NULL) {
    if (is.null(subfolder_name)) {
        output <- json_object$get_vector(json_vector_label, subfolder_name)
    } else {
        output <- json_object$get_vector(json_vector_label, subfolder_name)
    }
    return(output)
}

#' Load a scalar from json
#'
#' @param json_object Object of class `CppJson`
#' @param json_scalar_label Label referring to a particular scalar / string value (i.e. "num_samples") in the overall json hierarchy
#' @param subfolder_name (Optional) Name of the subfolder / hierarchy under which vector sits
#'
#' @return R vector
#' @export
loadScalarJson <- function(json_object, json_scalar_label, subfolder_name = NULL) {
    if (is.null(subfolder_name)) {
        output <- json_object$get_vector(json_scalar_label, subfolder_name)
    } else {
        output <- json_object$get_vector(json_scalar_label, subfolder_name)
    }
    return(output)
}

#' Create a new (empty) C++ Json object
#'
#' @return `CppJson` object
#' @export
createCppJson <- function() {
    return(invisible((
        CppJson$new()
    )))
}

#' Create a C++ Json object from a Json file
#'
#' @param json_filename Name of file to read. Must end in `.json`.
#' @return `CppJson` object
#' @export
createCppJsonFile <- function(json_filename) {
    invisible((
        output <- CppJson$new()
    ))
    output$load_from_file(json_filename)
    return(output)
}

#' Create a C++ Json object from a Json string
#'
#' @param json_string JSON string dump
#' @return `CppJson` object
#' @export
createCppJsonString <- function(json_string) {
    invisible((
        output <- CppJson$new()
    ))
    output$load_from_string(json_string)
    return(output)
}
