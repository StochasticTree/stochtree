# Docker for `stochtree` Development

Dockerfiles in their respective language directories (`R`, `Python`) set up a reproducible linux environment for testing `stochtree`.

All commands below should be run from the top-level `stochtree` project directory.

## R Development

### Github Source

To build the R Docker image from the latest github source:

```bash
docker build -t stochtree-r -f tools/docker/R/Dockerfile-R-Github .
```

To rebuild the image from scratch (after i.e. the github source has changed):

```bash
docker build --no-cache -t stochtree-r -f tools/docker/R/Dockerfile-R-Github .
```

### Local Source

To build the R Docker image from a local directory, first create the `stochtree_cran` directory with a properly formatted R package structure:

```bash
Rscript cran-bootstrap.R 0 0 1
```

(Run `Rscript cran-cleanup.R` if you have a previously-generated `stochtree_cran` subdirectory in your `stochtree` directory.)

Then build the Docker image:

```bash
docker build -t stochtree-r -f tools/docker/R/Dockerfile-R-Local stochtree_cran
```

### Running example scripts

To compare BART results on different platforms for a simple simulated dataset, run:

```bash
docker run --rm -v $(pwd):/workspace stochtree-r Rscript /workspace/tools/docker/R/bart_quick_check.R
```

To compare BCF results on different platforms for a simple simulated dataset, run:

```bash
docker run --rm -v $(pwd):/workspace stochtree-r Rscript /workspace/tools/docker/R/bcf_quick_check.R
```

## Python Development

### Github Source

To build the Python Docker image from the latest github source:

```bash
docker build -t stochtree-python -f tools/docker/Python/Dockerfile-Python-Github .
```

To rebuild the image from scratch (after i.e. the github source has changed):

```bash
docker build --no-cache -t stochtree-python -f tools/docker/Python/Dockerfile-Python-Github .
```

### Local Source

To build the Python Docker image from a local directory:

```bash
docker build -t stochtree-python -f tools/docker/Python/Dockerfile-Python-Local .
```

### Running example scripts

To compare BART results on different platforms for a simple simulated dataset, run:

```bash
docker run --rm -v $(pwd):/workspace stochtree-python python /workspace/tools/docker/Python/bart_quick_check.py
```

To compare BCF results on different platforms for a simple simulated dataset, run:

```bash
docker run --rm -v $(pwd):/workspace stochtree-python python /workspace/tools/docker/Python/bcf_quick_check.py
```
