import pandas as pd
from sklearn.model_selection import train_test_split

def get_data_split(df: pd.DataFrame, seed: int) -> tuple:
    """
    Split dataframe into training and testing sets (80/20).
    Returns DataFrames (not scaled arrays).
    """
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        stratify=y,
                                                        random_state=seed)
    return X_train, X_test, y_train, y_test