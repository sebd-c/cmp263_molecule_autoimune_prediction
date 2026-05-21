# Exploratory Data Analysis
from matplotlib.artist import get

from plotters import get_plots
from pre_processing import dataset_fix

path_to_og_dataset_test = "src/dataset/DIA_testset_RDKit_descriptors.csv"
path_to_og_dataset_train = "src/dataset/DIA_trainingset_RDKit_descriptors.csv"

df_og = dataset_fix.unite_dataset(path_to_og_dataset_test, path_to_og_dataset_train)
df_fix, dic_fix_vartypes, empty_instances, list_clones, list_homo = (
    dataset_fix.fix_dataset(df_og, remove_smiles=True)
)

get_plots.binary_att_proportions(df_fix, dic_fix_vartypes)
get_plots.boxplot_numerical(df_fix, dic_fix_vartypes, 4, ignore_attribut="Ipc")
