import pandas as pd
import pytest

from ml_model_utils.evaluation import (
    create_confusion_matrix, create_classification_report
)


def test_create_classification_report():
    result = create_classification_report([1, 2, 3, 4, 5], [2, 1, 3, 4, 5])
    expected_result = pd.DataFrame([[0.0, 0.0, 0.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0],
                                    [0.6, 0.6, 0.6, 0.6],
                                    [0.6, 0.6, 0.6, 5.0],
                                    [0.6, 0.6, 0.6, 5.0]],
                                   index=['1', '2', '3', '4', '5', "accuracy", 'macro avg', 'weighted avg'],
                                   columns=['precision', 'recall', 'f1-score', 'support'])
    pd.testing.assert_frame_equal(result, expected_result)


@pytest.mark.parametrize("y_test, y_pred, percentage, expected_result",
                         [([1, 2, 3, 4, 5], [2, 1, 3, 4, 5], False,
                           pd.DataFrame([[0, 1, 0, 0, 0],
                                         [1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0],
                                         [0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 1]],
                                        index=[1, 2, 3, 4, 5],
                                        columns=[1, 2, 3, 4, 5])),
                          ([1, 2, 3, 4, 5], [2, 1, 3, 4, 5], True,
                           pd.DataFrame([[0, 100.0, 0, 0, 0],
                                         [100.0, 0, 0, 0, 0],
                                         [0, 0, 100.0, 0, 0],
                                         [0, 0, 0, 100.0, 0],
                                         [0, 0, 0, 0, 100.0]],
                                        index=[1, 2, 3, 4, 5],
                                        columns=[1, 2, 3, 4, 5])),
                          (['1', '2', '3', '4', '5'], ['2', '1', '3', '4', '5'], False,
                           pd.DataFrame([[0, 1, 0, 0, 0],
                                         [1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0],
                                         [0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 1]],
                                        index=['1', '2', '3', '4', '5'],
                                        columns=['1', '2', '3', '4', '5']))])
def test_create_confusion_matrix(y_test, y_pred, percentage, expected_result):
    result = create_confusion_matrix(y_test, y_pred, percentage)
    pd.testing.assert_frame_equal(result, expected_result)
