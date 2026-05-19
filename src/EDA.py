#Exploratory Data Analysis
from pre_processing import dataset_fix
path_to_og_dataset_test = "src/dataset/DIA_testset_RDKit_descriptors.csv"
path_to_og_dataset_train = "src/dataset/DIA_trainingset_RDKit_descriptors.csv"

df_og=dataset_fix.unite_dataset(path_to_og_dataset_test, path_to_og_dataset_train)
df_fix, fix_vartypes, empty_instances, list_clones, list_homo = dataset_fix.fix_dataset(df_og)
