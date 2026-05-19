# imports
import matplotlib.pyplot as plt
import pandas as pd
##############################################################################################

def plot_cv_metrics(scores_df: pd.DataFrame) -> None:
    """
    Plot train and test cross-validation metrics as simple boxplots.
    """
    train_cols = ["train_recall", "train_precision", "train_accuracy"]
    test_cols = ["test_recall", "test_precision", "test_accuracy"]
    metric_labels = ["Recall", "Precision", "Accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    axes[0].boxplot(scores_df[train_cols], tick_labels=metric_labels, showmeans=True)
    axes[0].set_title("Train")
    axes[0].set_ylabel("Score")

    axes[1].boxplot(scores_df[test_cols], tick_labels=metric_labels, showmeans=True)
    axes[1].set_title("Test")

    plt.tight_layout()
    plt.show()

#TODO: fix tree plotting here

# def draw_tree()->None:
#     dot_data = StringIO()
#     export_graphviz(clf, out_file=dot_data,
#                     feature_names=feature_cols,
#                     class_names=class_names,
#                     filled=True, rounded=True,
#                     special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     graph.write_png('decision_tree.png')
#     pass