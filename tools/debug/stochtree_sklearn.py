# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from stochtree import StochTreeRegressor, StochTreeBinaryClassifier

# Generate data
n = 100
p = 10
rng = np.random.default_rng(42)
X = rng.normal(size=(n, p))
y = X[:, 0] * 3 + rng.normal(size=n)

# Fit and predict a model
reg1 = StochTreeRegressor(general_params={"random_seed": 42})
reg1.fit(X, y)
pred1 = reg1.predict(X)

# Check that we get the same results with the same seed
# Also check that we can run the model on pandas inputs
X_df = pd.DataFrame(X)
y_series = pd.Series(y)
reg2 = StochTreeRegressor(general_params={"random_seed": 42})
reg2.fit(X_df, y_series)
pred2 = reg2.predict(X_df)

# Compare the predictions of each model
plt.clf()
plt.scatter(pred1, pred2)
plt.xlabel("First model")
plt.ylabel("Second model")
plt.title("Comparison of Predictions")
plt.show()

# Check that StochTreeRegressor is a valid estimator
check_estimator(StochTreeRegressor(general_params={"random_seed": 42}, mean_forest_params={"min_samples_leaf": 1}))

# Check that can cross validate the "alpha" parameter of the mean forest
param_grid = {
    'num_gfr': [5, 10, 15],
    'num_mcmc': [50, 100, 200],
    'mean_forest_params': [
        {'num_trees': 50, 'alpha': 0.95, 'beta': 2.0},
        {'num_trees': 100, 'alpha': 0.90, 'beta': 1.5},
        {'num_trees': 200, 'alpha': 0.85, 'beta': 1.0}
    ]
}
grid_search = GridSearchCV(
    estimator=StochTreeRegressor(),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X, y)
# grid_search.cv_results_
# grid_search.best_estimator_

# Load a binary classification dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# Check that we can fit and predict on this dataset
clf = StochTreeBinaryClassifier(general_params={"random_seed": 42})
clf.fit(X=X, y=y)

# Load a multiclass classification dataset
dataset = load_wine()
X = dataset.data
y = dataset.target

# Check that we can fit and predict on this dataset by wrapping in the OneVsRest meta-estimator
clf = OneVsRestClassifier(StochTreeBinaryClassifier(general_params={"random_seed": 42}))
clf.fit(X=X, y=y)

# Check that we have a valid general purpose classifier when wrapping this estimator in the OneVsRest meta-estimator
check_estimator(OneVsRestClassifier(StochTreeBinaryClassifier(general_params={"random_seed": 42}, mean_forest_params={"min_samples_leaf": 1})))
