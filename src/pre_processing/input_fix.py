import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from pre_processing.dataset_fix import get_numeric_vartypes

class NumericalDiscretizer:
    """
    Transformer that fits discretization on training data and transforms test data.
    Prevents data leakage.
    """
    def __init__(self, vartypes_dic, method='quantile', n_bins=4):
        self.vartypes_dic = vartypes_dic
        self.method = method
        self.n_bins = n_bins
        self.discretizers = {}  # Store fitted discretizers per column
        self.binary_flags = {}  # Store which columns are binary flags
        self.constant_cols = {}  # Store constant columns info
        
    def fit(self, df):
        numerical_att_dic = get_numeric_vartypes(df, self.vartypes_dic)
        
        for att in numerical_att_dic:
            data = df[att].values
            unique_n = df[att].nunique()
            zero_ratio = (df[att] == 0).sum() / len(df)
            
            # Check for sparse columns
            if zero_ratio > 0.9 and unique_n < 10:
                self.binary_flags[att] = True
                continue
            
            actual_bins = unique_n if unique_n < self.n_bins else self.n_bins
            
            if numerical_att_dic[att] == 'int':
                if unique_n < self.n_bins:
                    bins = np.arange(data.min(), data.max() + 2, 
                                    (data.max() - data.min() + 1) / actual_bins)
                    self.discretizers[att] = ('cut', bins)
                else:
                    try:
                        if self.method == 'quantile':
                            binned, bins = pd.qcut(data, q=actual_bins, labels=False, 
                                                   retbins=True, duplicates='drop')
                        else:
                            binned, bins = pd.cut(data, bins=actual_bins, labels=False, 
                                                  retbins=True, duplicates='drop')
                        self.discretizers[att] = ('cut', bins)
                    except ValueError:
                        binned, bins = pd.cut(data, bins=actual_bins, labels=False, 
                                              retbins=True, duplicates='drop')
                        self.discretizers[att] = ('cut', bins)
                
                # Check if we got only 1 bin
                temp_binned = pd.cut(data, bins=self.discretizers[att][1], labels=False, include_lowest=True)
                if len(np.unique(temp_binned)) < 2:
                    self.binary_flags[att] = True
                    del self.discretizers[att]
            
            elif numerical_att_dic[att] == 'float':
                data_reshaped = data.reshape(-1, 1)
                actual_bins = min(actual_bins, unique_n)
                
                if actual_bins >= 2:
                    try:
                        discretizer = KBinsDiscretizer(n_bins=actual_bins, encode='ordinal', 
                                                      strategy=self.method)
                        discretizer.fit(data_reshaped)
                        self.discretizers[att] = ('kbins', discretizer)
                        
                        # Check if we got only 1 bin
                        temp_binned = discretizer.transform(data_reshaped)
                        if len(np.unique(temp_binned)) < 2:
                            self.binary_flags[att] = True
                            del self.discretizers[att]
                    except Exception:
                        self.binary_flags[att] = True
                else:
                    self.binary_flags[att] = True
        
        return self
    
    def transform(self, df):
        df_result = df.copy()
        bin_columns = {}
        
        for att in self.discretizers:
            data = df[att].values
            
            if self.discretizers[att][0] == 'cut':
                bins = self.discretizers[att][1]
                binned = pd.cut(data, bins=bins, labels=False, include_lowest=True)
                bin_columns[att + '_bin'] = binned
            else:  # kbins
                discretizer = self.discretizers[att][1]
                data_reshaped = data.reshape(-1, 1)
                binned = discretizer.transform(data_reshaped)
                bin_columns[att + '_bin'] = binned.flatten()
        
        for att in self.binary_flags:
            binned = (df[att] > 0).astype(int)
            bin_columns[att + '_bin'] = binned
        
        if bin_columns:
            bin_df = pd.DataFrame(bin_columns)
            df_result = pd.concat([df_result, bin_df], axis=1)
        
        return df_result
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)