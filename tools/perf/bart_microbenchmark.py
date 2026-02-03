import timeit

# Translate the R code above to python
setup_code = """
import numpy as np
from stochtree import BARTModel
rng = np.random.default_rng()
n = 1000
p = 20
X = rng.uniform(0, 1, (n, p))
f_XW = np.where((0 <= X[:, 0]) & (0.25 > X[:, 0]), -7.5,
                 np.where((0.25 <= X[:, 0]) & (0.5 > X[:, 0]), -2.5,
                          np.where((0.5 <= X[:, 0]) & (0.75 > X[:, 0]), 2.5,
                                   np.where((0.75 <= X[:, 0]) & (1 > X[:, 0]), 7.5, 0))))
y = f_XW + rng.normal(0, 1, n)
"""

num_times = 10
time_bart = timeit.timeit("""
bart_model = BARTModel()
bart_model.sample(X, y, num_gfr=10, num_mcmc=100)
""", setup=setup_code, number=num_times)
print(f"Average runtime of {time_bart / num_times:<.6f} seconds")
