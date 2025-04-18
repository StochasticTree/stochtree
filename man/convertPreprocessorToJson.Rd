% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{convertPreprocessorToJson}
\alias{convertPreprocessorToJson}
\title{Convert the persistent aspects of a covariate preprocessor to (in-memory) C++ JSON object}
\usage{
convertPreprocessorToJson(object)
}
\arguments{
\item{object}{List containing information on variables, including train set
categories for categorical variables}
}
\value{
wrapper around in-memory C++ JSON object
}
\description{
Convert the persistent aspects of a covariate preprocessor to (in-memory) C++ JSON object
}
\examples{
cov_mat <- matrix(1:12, ncol = 3)
preprocess_list <- preprocessTrainData(cov_mat)
preprocessor_json <- convertPreprocessorToJson(preprocess_list$metadata)
}
