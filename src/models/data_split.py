# imports
import pandas as pd
from sklearn.model_selection import RepeatedKFold, StratifiedKFold, train_test_split, cross_validate, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
#########################################################################################
def get_data_split(df) -> tuple:
    selected_features = []
    # separate trainable features and
    X = df[df[selected_features]]
    y = df['label']

    # separate between train/val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,
                                                                test_size=0.20,
                                                                stratify=y,
                                                                random_state=42)

    # separate again for train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.10,
                                                      stratify=y_train_val,
                                                      random_state=42)

    #normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

