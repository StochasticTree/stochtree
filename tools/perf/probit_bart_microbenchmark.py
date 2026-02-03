import timeit
import numpy as np
from stochtree import BARTModel, OutcomeModel

# Translate the R code above to python
setup_code = """
import numpy as np
rng = np.random.default_rng()
n = 1000
p = 20
X = rng.uniform(0, 1, (n, p))
f_XW = np.where((0 <= X[:, 0]) & (0.25 > X[:, 0]), -7.5,
                 np.where((0.25 <= X[:, 0]) & (0.5 > X[:, 0]), -2.5,
                          np.where((0.5 <= X[:, 0]) & (0.75 > X[:, 0]), 2.5,
                                   np.where((0.75 <= X[:, 0]) & (1 > X[:, 0]), 7.5, 0))))
z = f_XW + rng.normal(0, 1, n)
y = (z > 0).astype(int)
"""

time_probit = timeit.timeit("""
from stochtree import BARTModel, OutcomeModel
general_params = {'outcome_model': OutcomeModel(outcome='binary', link='probit'), 
                  'sample_sigma2_global': False}
bart_model = BARTModel()
bart_model.sample(X, y, num_gfr=10, num_mcmc=100, general_params=general_params)
""", setup=setup_code, number=10)
