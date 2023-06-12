import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from typing import Optional, List, Any, Sequence
from .evaluation import create_confusion_matrix


def distribution_hist(data: pd.DataFrame,
                      col: str,
                      figure_size: int,
                      output_path: Optional[str] = None):
    """ Histogram distribution of the column in the given data.

    Args:
        data (:obj:`pandas.DataFrame`): dataset.
        col (str): target column for the distribution to be observed.
        figure_size (int): scale for the distribution plot.
        output_path (str, optional): output image file path. If no output path
          is given, instead of saving, plt.show() is executed.

    Examples:
        >>> distribution_hist(df, "target", 2)
    """
    plt.figure(figsize=(figure_size * 4, figure_size * 2))
    sns.barplot(x=data[col].value_counts().index, y=data[col].value_counts().values,
                alpha=0.8)
    plt.title(f"Distribution of the column {col}")
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.xticks(rotation=90)
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_confusion_matrix(y_test: Sequence[Any],
                          y_pred: Sequence[Any],
                          figure_size: int,
                          percentage: bool = False,
                          selected_labels: Optional[List[Any]] = None):
    """ Confusion matrix plot of given test & preds using seaborn library.

    Args:
        y_test (:obj:`list` of any): labels in test data.
        y_pred (:obj:`list` of any): predictions for the test data.
        figure_size (int): scale for the confusion matrix plot
        percentage (bool, optional): boolean value that decides whether over the scale 100 or
          exact numbers will be used in the confusion matrix. Default is False.
        selected_labels (:obj:`list` of :obj:`str`, optional): selected labels to slice the
          confusion matrix

    Examples:
        >>> plot_confusion_matrix(test, pred, "target", 2)
    """
    df_cm = create_confusion_matrix(y_test, y_pred, percentage, selected_labels)
    plt.figure(figsize=(4 * figure_size, 3 * figure_size))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
