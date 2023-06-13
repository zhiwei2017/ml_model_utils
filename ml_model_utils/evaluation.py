import numpy as np
import pandas as pd  # type: ignore
from sklearn.metrics import confusion_matrix, classification_report  # type: ignore
from typing import Optional, List, Any, Sequence


def create_classification_report(y_test: Sequence[Any],
                                 y_pred: Sequence[Any]) -> pd.DataFrame:
    """Create a classification report and convert it to pandas DataFrame format.

     Args:
        y_test (:obj:`list` of any): labels in test data.
        y_pred (:obj:`list` of any): predictions for the test data.

    Returns:
        :obj:`pandas.DataFrame`: classification report having precision, recall,
        f1-score, support columns

    Examples:
        >>> create_classification_report([1, 2, 3, 4, 5], [2, 1, 3, 4, 5])
                      precision  recall  f1-score  support
        1                   0.0     0.0       0.0      1.0
        2                   0.0     0.0       0.0      1.0
        3                   1.0     1.0       1.0      1.0
        4                   1.0     1.0       1.0      1.0
        5                   1.0     1.0       1.0      1.0
        accuracy            0.6     0.6       0.6      0.6
        macro avg           0.6     0.6       0.6      5.0
        weighted avg        0.6     0.6       0.6      5.0
    """
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    return df_report


def create_confusion_matrix(y_test: Sequence[Any],
                            y_pred: Sequence[Any],
                            percentage: bool = False,
                            selected_labels: Optional[List[Any]] = None) -> pd.DataFrame:
    """Create confusion matrix in pandas DataFrame format.

    Args:
        y_test (:obj:`list` of any): labels in test data.
        y_pred (:obj:`list` of any): predictions for the test data.
        percentage (bool, optional): use percentage as the ceil value. Default is False.
        selected_labels (:obj:`list` of any, optional): selected labels for the confusion matrix.
          If none, the confusion matrix dataframe will use all labels.

    Returns:
        :obj:`pandas.DataFrame`: confusion matrix in pandas DataFrame format.

    Examples:
        >>> create_confusion_matrix([1, 2, 3, 4, 5], [2, 1, 3, 4, 5])
           1  2  3  4  5
        1  0  1  0  0  0
        2  1  0  0  0  0
        3  0  0  1  0  0
        4  0  0  0  1  0
        5  0  0  0  0  1
    """
    labels = sorted(selected_labels if selected_labels else set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    if percentage:
        cm = np.round(100 * cm / np.sum(cm, axis=1).reshape(-1, 1))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return cm_df
