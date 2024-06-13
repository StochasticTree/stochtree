import numpy as np
import pandas as pd
from stochtree import CovariateTransformer

class TestPreprocessor:
    def test_numpy(self):
        cov_transformer = CovariateTransformer()
        np_1 = np.array(
            [[1.5, 8.7, 1.2],
             [2.7, 3.4, 5.4],
             [3.6, 1.2, 9.3],
             [4.4, 5.4, 10.4],
             [5.3, 9.3, 3.6],
             [6.1, 10.4, 4.4]]
        )
        np_1_transformed = cov_transformer.fit_transform(np_1)
        np.testing.assert_array_equal(np_1, np_1_transformed)
        assert cov_transformer._processed_feature_types == [0,0,0]

    def test_pandas(self):
        cov_transformer = CovariateTransformer()
        df_1 = pd.DataFrame(
            {"x1": [1.5, 2.7, 3.6, 4.4, 5.3, 6.1],
             "x2": [8.7, 3.4, 1.2, 5.4, 9.3, 10.4],
             "x3": [1.2, 5.4, 9.3, 10.4, 3.6, 4.4]}
        )
        np_1 = np.array(
            [[1.5, 8.7, 1.2],
             [2.7, 3.4, 5.4],
             [3.6, 1.2, 9.3],
             [4.4, 5.4, 10.4],
             [5.3, 9.3, 3.6],
             [6.1, 10.4, 4.4]]
        )
        df_1_transformed = cov_transformer.fit_transform(df_1)
        np.testing.assert_array_equal(np_1, df_1_transformed)
        assert cov_transformer._processed_feature_types == [0,0,0]
