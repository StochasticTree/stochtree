## R CMD check results

0 errors | 0 warnings | 2 notes

* Checking installed package size ... NOTE installed size is 46.3Mb (linux-only)
* Possibly misspelled words in DESCRIPTION: All of the words are proper nouns or technical terms (BCF, Carvalho, Chipman, McCulloch, XBART)

## CRAN comments (20250206)

Below are responses to the initial comments received from CRAN on Feb 6, 2025

### Copyright

> Please always add all authors, contributors and copyright holders in the Authors@R field with the appropriate roles."

stochtree's C++ core has several vendored dependencies. The license and copyright details for each of these dependencies are delineated in the inst/COPYRIGHTS file. We have included the authors / contributors of each of these dependencies as copyright holders in the authors list of the DESCRIPTION file and also included a "Copyright:" section in the DESCRIPTION file explaining this.

### TRUE / FALSE

> Please write TRUE and FALSE instead of T and F.

We have converted `T` and `F` to `TRUE` and `FALSE` in the R code.

### Examples with commented code

> Some code lines in examples are commented out. Please never do that.

We no longer do this, and apologize for the oversight.

## CRAN comments (20250207)

Below we address issues raised by CRAN on Feb 7, 2025

### Valgrind

A valgrind-instrumented version of R exposed memory issues in several examples 
in the `stochtree` documentation. The specific issue is 

> Conditional jump or move depends on uninitialised value(s)

The examples that triggered this were in fact working with Eigen matrices 
with uninitialized values. 

This has been corrected and we have verified that running the `stochtree` 
examples no longer produce this memcheck error.
