# imports

##############################################################################################

def draw_tree()->None:
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    feature_names=feature_cols,
                    class_names=class_names,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('decision_tree.png')
    pass