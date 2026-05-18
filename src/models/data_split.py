# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#########################################################################################
def get_data_split(df: pd.DataFrame,
                   seed: int
                   ) -> tuple:
    """
    Given a pandas dataframe, this function splits the dataframe into training and testing sets.
    :param df:
    :param seed:
    :return:
    """
    selected_features = []
    # separate trainable features and
    X = df[df[selected_features]]
    y = df['Label']

    # separate between train/val and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        stratify=y,
                                                        random_state=seed)

    #normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

