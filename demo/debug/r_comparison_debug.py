# R Comparison Demo Script

# Load necessary libraries
import numpy as np
import pandas as pd
from stochtree import BARTModel

# Load data
df = pd.read_csv("debug/data/heterosked_train.csv")
y = df.loc[:,'y'].to_numpy()
X = df.loc[:,['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10']].to_numpy()
y = y.astype(np.float64)
X = X.astype(np.float64)

# Run BART
bart_model = BARTModel()
bart_model.sample(X_train=X, y_train=y, num_gfr=0, num_mcmc=10, general_params={'random_seed': 1234, 'standardize': False, 'sample_sigma2_global': True})

# Inspect the MCMC (BART) samples
y_avg_mcmc = np.squeeze(bart_model.y_hat_train).mean(axis = 1, keepdims = True)
print(y_avg_mcmc[:20])
print(bart_model.global_var_samples)
