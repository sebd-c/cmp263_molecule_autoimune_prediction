# imports
from pandas import read_pickle
from pandas import DataFrame
import graphviz
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree
from six import StringIO
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
##############################################################################################




def get_metrics(y_train, y_pred) -> dict:
    """
    Returns dictionary of usual Machine Learning classification metrics
    """
    # get accuracies
    accuracy = accuracy_score(y_train, y_pred)
    bal_accuracy = balanced_accuracy_score(y_train, y_pred)

    # get precisions
    global_precision = precision_score(y_train, y_pred, average='micro')
    weighted_precision = precision_score(y_train, y_pred, average='weighted')

    # get recalls
    global_recall = recall_score(y_train, y_pred, average='micro')
    weighted_recall = recall_score(y_train, y_pred, average='weighted')

    # get f1-score
    global_f1 = f1_score(y_train, y_pred, average='micro')
    weighted_f1 = f1_score(y_train, y_pred, average='weighted')

    metrics_dict = {'accuracy': accuracy,
                    'bal_accuracy': bal_accuracy,
                    'global_precision': global_precision,
                    'weighted_precision': weighted_precision,
                    'global_recall': global_recall,
                    'weighted_recall': weighted_recall,
                    'global_f1_score': global_f1,
                    'weighted_f1_score': weighted_f1}
    return metrics_dict


def get_confusion_matrix(y_train, y_pred) -> None:
    """
    Plots confusion matrix of a classification model.
    """

    # define class names
    class_names = ['DIA-', 'DIA+']

    fig, ax = plt.subplots(figsize=(25, 25))
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred, ax=ax, normalize='true')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    _ = ax.set_title('Classification Confusion Matrix Train Normalized')

    plt.show()
    pass
