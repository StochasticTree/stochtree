% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/forest.R
\name{createForestContainer}
\alias{createForestContainer}
\title{Create a container of forest samples}
\usage{
createForestContainer(
  num_trees,
  output_dimension = 1,
  is_leaf_constant = F,
  is_exponentiated = F
)
}
\arguments{
\item{num_trees}{Number of trees}

\item{output_dimension}{Dimensionality of the outcome model}

\item{is_leaf_constant}{Whether leaf is constant}

\item{is_exponentiated}{Whether forest predictions should be exponentiated before being returned}
}
\value{
\code{ForestSamples} object
}
\description{
Create a container of forest samples
}
