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

def grouping_numeric(df, vartypes_dic, groups):
    numerical_cols = []
    for col in vartypes_dic:
        if vartypes_dic[col] == "numerical":
            numerical_cols.append(col)
    
    if len(numerical_cols) == 0:
        return []
    
    attr_ranges = []
    for col in numerical_cols:
        data = df[col].dropna()
        if len(data) > 0:
            amplitude = data.max() - data.min()
            attr_ranges.append((col, amplitude))
        else:
            attr_ranges.append((col, 0))
    
    attr_ranges.sort(key=lambda x: x[1])
    
    n_attrs = len(attr_ranges)
    group_size = n_attrs // groups
    
    numeric_groups = []
    start = 0
    
    for i in range(groups):
        if i == groups - 1:
            end = n_attrs
        else:
            end = start + group_size
        
        group_attrs = [attr[0] for attr in attr_ranges[start:end]]
        numeric_groups.append(group_attrs)
        start = end
        
    return numeric_groups


def boxplot_numerical(df, vartypes_dic, groups_n,ignore_attribut=None):
    numeric_groups = grouping_numeric(df, vartypes_dic, groups_n)
    if ignore_attribut is not None:
        # remove the ignore attribute from the group
        numeric_groups = [[attr for attr in group if attr != ignore_attribut] for group in numeric_groups]
    
    for i, group in enumerate(numeric_groups):
        figsize = (max(12, len(group) * 0.5), 6)
        fig, ax = plt.subplots(figsize=figsize)
        
        df[group].boxplot(ax=ax, rot=90, fontsize=8)
        
        ax.set_title(f"Boxplot - Grupo {i+1} ({len(group)} atributos)", fontsize=14)
        ax.set_xlabel("Atributos", fontsize=12)
        ax.set_ylabel("Valores", fontsize=12)
        
        plt.tight_layout()
        plt.show()

def plot_binary_heatmap(df, vartypes_dic):
    binary_cols = [col for col in vartypes_dic if vartypes_dic[col] == "binary"]
    
    # Calculate proportion of 1's for each variable
    proportions = df[binary_cols].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    proportions.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Proportion of 1s')
    ax.set_title(f'Distribution of {len(binary_cols)} Binary Variables')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return proportions