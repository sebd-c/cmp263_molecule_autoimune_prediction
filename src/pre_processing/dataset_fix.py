from dis import disco
import pandas as pd

def save_dataset(df,path="src/dataset/dataset.csv"):
        df.to_csv(path)

def unite_dataset(path_to_dataset1="", path_to_dataset2=""):
    df1 = pd.read_csv(path_to_dataset1)
    df2 = pd.read_csv(path_to_dataset2)
    df = pd.concat([df1, df2])
    #df = df.drop(df.columns[0], axis=1)
    df.reset_index(drop=True, inplace=True)

    save_dataset(df,path="src/dataset/pre_fix_dataset.csv")
    return df

def get_vartypes(df):
    dic_vartypes = dict()
    n_binaries=0
    n_categorical=0
    n_numerical=0
    # defining the data type for each column
    for col in df.columns:
        # binaries
        if df[col].nunique() == 2:
            dic_vartypes[col] = "binary"
            n_binaries += 1
        # categorial
        elif df[col].dtype == "object":
            dic_vartypes[col] = "categorical"
            n_categorical += 1
        # numerical
        elif df[col].dtype == "float64" or df[col].dtype == "int64":
            dic_vartypes[col] = "numerical"
            n_numerical += 1
        else:
            print(f"Column CHECK {col}: {df[col].dtype}")
            dic_vartypes[col] = "CHECK"
    print(f"number of atributes -label: {len(df.columns)-1}") 
    print(f"binaries -label: {(n_binaries)-1}, categorical: {n_categorical}, numerical: {n_numerical}")
    return dic_vartypes


def filter_homog_att(df):
    df_filtered = df.copy()
    n_homo=0
    list_homo=[]
    for col in df.columns:
        if df[col].nunique() == 1:
            df_filtered = df_filtered.drop(col, axis=1)
            n_homo += 1
            list_homo.append(col)
            
    
    print(f"Removed {n_homo} homogenous columns: {list_homo}")
    
    return df_filtered, list_homo

def filter_clones_att(df):
    df_filtered = df.copy()
    n_clones=0
    list_clones=[]
    for col in df.columns:
        if list(df.columns).count(col)>1:
            list_clones.append(col)
            n_clones += 1
    print(f"Found {n_clones} clones: {list_clones}")
    df_filtered = df_filtered.drop(list_clones, axis=1)
    return df_filtered, list_clones

def find_empty_instances(df):
    empty_instances = df[df.isnull().any(axis=1)]
    print(f"found {len(empty_instances)} empty instances")
    return empty_instances

def fix_dataset(df,remove_smiles=False):
    df_fixed = df.copy()

    #original vartypes 
    print("original vartypes")
    original_vartypes = get_vartypes(df_fixed)
    
    if remove_smiles:
        df_fixed = df_fixed.drop("SMILES", axis=1)
    
    #empty instances
    empty_instances = find_empty_instances(df_fixed)
    
    if not len(empty_instances)>0:
        df_fixed = df_fixed.drop(empty_instances.index, axis=0)

    #clones attriubtes
    
    df_fixed, list_clones = filter_clones_att(df_fixed)

    
    # homogenous attributes
    df_fixed, list_homo = filter_homog_att(df_fixed)

    # pos-fix vartypes
    print("fixed vartypes")
    fix_vartypes = get_vartypes(df_fixed)
    
    return df_fixed, fix_vartypes, empty_instances, list_clones, list_homo


