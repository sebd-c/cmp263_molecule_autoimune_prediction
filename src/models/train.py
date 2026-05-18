# imports
import pandas as pd
#########################################################################################
def run_decision_tree(data: pd.DataFrame) -> None:

    """
    Given a dataframe of tabular instances, run decision tree
    """
    # set columns of interest
    feature_cols = []

    label_dict = {0: 'DIA-', 1: 'DIA+'}

    data.replace({'Label': label_dict}, inplace=True)

    # drop the ground truth column
    X = data.drop(['label'], axis=1)
    X = X[feature_cols]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

    # class names
    class_names = ['Normal', 'Quiescent', 'Fully Senescent', 'Senescent-like']

    # train model
    clf = DecisionTreeClassifier(criterion='entropy',
                                 random_state=0,
                                 max_depth=7,
                                 ccp_alpha=0.019,
                                 min_samples_leaf=10,
                                 max_features="sqrt",
                                 class_weight='balanced')

    # fit the model
    clf.fit(X_train, y_train)

    # repeat accuracy for train to check for overfitting
    y_pred_train = clf.predict(X_train)

    # metrics for train set