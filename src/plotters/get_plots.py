# imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

##############################################################################################


def draw_tree() -> None:
    dot_data = StringIO()
    export_graphviz(
        clf,
        out_file=dot_data,
        feature_names=feature_cols,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("decision_tree.png")
    pass


def binary_att_proportions(df, vartypes_dic):
    binary_cols = []
    for col in vartypes_dic:
        if vartypes_dic[col] == "binary":
            binary_cols.append(col)

    proportions = []
    for col in binary_cols:
        count = df[col].value_counts(normalize=True)
        prop_1 = count.get(1, 0)
        prop_0 = count.get(0, 0)
        proportions.append((prop_0, prop_1))

    plot_df = pd.DataFrame(proportions, index=binary_cols, columns=["0", "1"])
    fig, ax = plt.subplots(figsize=(12, max(6, len(binary_cols) * 0.3)))
    plot_df.plot(kind="barh", stacked=True, ax=ax, color=["red", "green"])

    ax.set_xlabel("Proportion")
    ax.set_ylabel("Feature")
    ax.set_title("Binarty atts proprotions")
    ax.legend(["0", "1"])

    for i, (idx, row) in enumerate(plot_df.iterrows()):
        if row["0"] > 0.05:
            ax.text(row["0"] / 2, i, f"{row['0']:.1%}", ha="center", va="center")
        if row["1"] > 0.05:
            ax.text(
                row["0"] + row["1"] / 2, i, f"{row['1']:.1%}", ha="center", va="center"
            )

    return plt.show()


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


# TODO: fix tree plotting here

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
