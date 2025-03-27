import numpy as np
import pandas as pd

from stochtree import CovariatePreprocessor


class TestPreprocessor:
    def test_numpy(self):
        cov_transformer = CovariatePreprocessor()
        np_1 = np.array(
            [
                [1.5, 8.7, 1.2],
                [2.7, 3.4, 5.4],
                [3.6, 1.2, 9.3],
                [4.4, 5.4, 10.4],
                [5.3, 9.3, 3.6],
                [6.1, 10.4, 4.4],
            ]
        )
        np_1_transformed = cov_transformer.fit_transform(np_1)
        np.testing.assert_array_equal(np_1, np_1_transformed)
        np.testing.assert_array_equal(
            cov_transformer._processed_feature_types, np.array([0, 0, 0])
        )

    def test_pandas(self):
        df_1 = pd.DataFrame(
            {
                "x1": [1.5, 2.7, 3.6, 4.4, 5.3, 6.1],
                "x2": [8.7, 3.4, 1.2, 5.4, 9.3, 10.4],
                "x3": [1.2, 5.4, 9.3, 10.4, 3.6, 4.4],
            }
        )
        np_1 = np.array(
            [
                [1.5, 8.7, 1.2],
                [2.7, 3.4, 5.4],
                [3.6, 1.2, 9.3],
                [4.4, 5.4, 10.4],
                [5.3, 9.3, 3.6],
                [6.1, 10.4, 4.4],
            ]
        )
        cov_transformer = CovariatePreprocessor()
        df_1_transformed = cov_transformer.fit_transform(df_1)
        np.testing.assert_array_equal(np_1, df_1_transformed)
        np.testing.assert_array_equal(
            cov_transformer._processed_feature_types, np.array([0, 0, 0])
        )

        df_2 = pd.DataFrame(
            {
                "x1": [1.5, 2.7, 3.6, 4.4, 5.3, 6.1],
                "x2": pd.Categorical(
                    ["a", "b", "c", "a", "b", "c"],
                    ordered=True,
                    categories=["c", "b", "a"],
                ),
                "x3": [1.2, 5.4, 9.3, 10.4, 3.6, 4.4],
            }
        )
        np_2 = np.array(
            [
                [1.5, 2, 1.2],
                [2.7, 1, 5.4],
                [3.6, 0, 9.3],
                [4.4, 2, 10.4],
                [5.3, 1, 3.6],
                [6.1, 0, 4.4],
            ]
        )
        cov_transformer = CovariatePreprocessor()
        df_2_transformed = cov_transformer.fit_transform(df_2)
        np.testing.assert_array_equal(np_2, df_2_transformed)
        np.testing.assert_array_equal(
            cov_transformer._processed_feature_types, np.array([0, 1, 0])
        )

        df_3 = pd.DataFrame(
            {
                "x1": [1.5, 2.7, 3.6, 4.4, 5.3, 6.1],
                "x2": pd.Categorical(
                    ["a", "b", "c", "a", "b", "c"],
                    ordered=False,
                    categories=["c", "b", "a"],
                ),
                "x3": [1.2, 5.4, 9.3, 10.4, 3.6, 4.4],
            }
        )
        np_3 = np.array(
            [
                [1.5, 0, 0, 1, 1.2],
                [2.7, 0, 1, 0, 5.4],
                [3.6, 1, 0, 0, 9.3],
                [4.4, 0, 0, 1, 10.4],
                [5.3, 0, 1, 0, 3.6],
                [6.1, 1, 0, 0, 4.4],
            ]
        )
        cov_transformer = CovariatePreprocessor()
        df_3_transformed = cov_transformer.fit_transform(df_3)
        np.testing.assert_array_equal(np_3, df_3_transformed)
        np.testing.assert_array_equal(
            cov_transformer._processed_feature_types, np.array([0, 1, 1, 1, 0])
        )

        df_4 = pd.DataFrame(
            {
                "x1": [1.5, 2.7, 3.6, 4.4, 5.3, 6.1, 7.6],
                "x2": pd.Categorical(
                    ["a", "b", "c", "a", "b", "c", "c"],
                    ordered=False,
                    categories=["c", "b", "a", "d"],
                ),
                "x3": [1.2, 5.4, 9.3, 10.4, 3.6, 4.4, 3.4],
            }
        )
        np_4 = np.array(
            [
                [1.5, 0, 0, 1, 0, 1.2],
                [2.7, 0, 1, 0, 0, 5.4],
                [3.6, 1, 0, 0, 0, 9.3],
                [4.4, 0, 0, 1, 0, 10.4],
                [5.3, 0, 1, 0, 0, 3.6],
                [6.1, 1, 0, 0, 0, 4.4],
                [7.6, 1, 0, 0, 0, 3.4],
            ]
        )
        cov_transformer = CovariatePreprocessor()
        df_4_transformed = cov_transformer.fit_transform(df_4)
        np.testing.assert_array_equal(np_4, df_4_transformed)
        np.testing.assert_array_equal(
            cov_transformer._processed_feature_types, np.array([0, 1, 1, 1, 1, 0])
        )
