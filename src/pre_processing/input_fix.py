import pandas as pd
import numpy as np
from scipy.integrate import newton_cotes
from sklearn.preprocessing import KBinsDiscretizer

def discretize_numerical_att(df,vartypes_dic, method='quantile', n_bins=4):

    list_numerical_att=[]
    for att in vartypes_dic:
        if vartypes_dic[att] =='numerical':
            list_numerical_att.append(att)

    bin_columns={}
    for att in list_numerical_att:
        
        data=df[att].values.reshape(-1,1)
        unique_n=df[att].nunique()
        if  unique_n < n_bins:
            small_bin=unique_n
            discretizer= KBinsDiscretizer(n_bins=small_bin, encode='ordinal', strategy=method)
            binned=discretizer.fit_transform(data)
            bin_columns[att+'_bin'] = binned
        else:
            discretizer= KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=method)
            binned=discretizer.fit_transform(data)
            bin_columns[att+'_bin'] = binned
        
        print(att)
        print("bin edges:", discretizer.bin_edges_[0])
        print("bin values:", np.unique(binned))
        print("-" * 50)

    return df

