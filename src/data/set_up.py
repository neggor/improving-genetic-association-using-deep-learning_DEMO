# IMPORT MODULES
import os
import sys
import subprocess
import argparse
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm

cwd = os.getcwd()

sys.path.insert(0, cwd) #add base to path for relative imports


from src.data.data_handler import create_dataset, generate_PCA_pairs
from src.features.process_data import split_train_val_test

from sklearn.model_selection import GroupShuffleSplit 


parser = argparse.ArgumentParser()

parser.add_argument('-dr', type = int,
    help = '0 or 1, Download raw data from kaggle', required= True)

parser.add_argument('-dc', type = int,
    help = '0 or 1, Download clean data from kaggle. \
  If 0 it will generate the data from raw.', required= True)

#parser.add_argument('-gs', type = int,
#    help = '0 or 1, Stratify train/test/val by genotype', required= True)

parser.add_argument('-pca', type = int,
    help = '0 or 1, generate PCA response values', required= True)

parser.add_argument('-snp', type = int,
    help = 'SNP (R position) used as Y', required= True)


args = parser.parse_args()
my_args = {k:v for k,v in args._get_kwargs() if v is not None}
my_selcted_SNP = my_args['snp']
#Stratify = my_args['gs']

#print(args.donw_raw) # Construct clean from raw
#print(args.donw_clean) # Construct clean from raw

origin = os.path.abspath('')

# Set environment variable to allow download from Kaggle everywhere
os.environ['KAGGLE_USERNAME'] = ''
os.environ['KAGGLE_KEY'] = ''
assert not \
    (os.environ['KAGGLE_USERNAME'] == '' or os.environ['KAGGLE_KEY'] == ''), \
    'Kaggle API keys needed. Request access!'
print('Kaggle API keys added!')
## Data criteria:
# Do not downsample original length
Downsampling_factor = 1
# Maximum of consecutive missing steps allowed
Max_n = 150
# Amount of steps reduced from the end of the dataset after moving
#  arenas with missing values at the beginning
Max_e = 500 
# value in mm sec for speed to be considered a missing value
Outlier_threshold = 20

######################################## DOWNLOAD RAW DATA

if my_args['dr']:
    if 'raw_data' in os.listdir('data/raw'):
        print('Raw data already present!')
    else:
        print('DOWNLOADING RAW DATA')
        os.chdir('data/raw')
        subprocess.run('kaggle datasets download -d jordialonsoesteve/raw-data',
        shell = True)
        #os.system('unzip raw-data.zip')
        with zipfile.ZipFile('raw-data.zip', 'r') as zip_ref:
            zip_ref.extractall('.')

        #subprocess.run(['unzip', 'raw-data.zip'])
        subprocess.run('rm raw-data.zip', shell=True)
        os.chdir(origin)
else:
    if 'raw_data' not in os.listdir('data/raw'):
        print('Data not available. Try to download it first!')
        exit()
######################################## 

######################################## DOWNLOAD CLEAN DATA
if my_args['dc']:
    os.chdir('data/interim')
    subprocess.run('kaggle datasets download -d jordialonsoesteve/interim-data',
    shell= True)
    with zipfile.ZipFile('interim-data.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    subprocess.run('rm interim-data.zip', shell= True)
    os.chdir(origin)

else:
    if 'featured_trajectories' in os.listdir('data/interim'): 
        # Which means that there should be no folders! 
        print('Data already present!')
    else:
        # Criteria to include in feature engineering
        Derivatives = True
        Spe_add = True
        Acc_add = True
        Ang_speed = True
        Trav_dist = True 
        Str_line = False
        snps = [85999,
                86000,
                86001,
                86002,
                86003,
                86004,
                86005,
                86006,
                12233,
                82727,
                100200] # Markers for which we create response.

        if my_selcted_SNP not in snps:
            print('NON-VALID SNP')
            exit()

        # This will populate the corresponding folders with the clean data: 
        # Cleaned and featured trajectories, response and genotype.
        os.chdir('data/interim')
        subprocess.run('mkdir featured_trajectories', shell=True)
        subprocess.run('mkdir Genotype_class', shell=True)
        subprocess.run('mkdir SNPs', shell=True)
        subprocess.run('mkdir missing_data_info', shell=True)

        os.chdir(origin)

    # Run, this takes a while
        create_dataset(snps,
                    Max_n,
                    Max_e,
                    Outlier_threshold,
                    Downsampling_factor,
                    Derivatives,
                    Spe_add,
                    Ang_speed,
                    Trav_dist,
                    Str_line, 
                    Acc_add) # Stores in interim
    ########################################
#exit()

######################################## SPLIT TRAIN/VAL/TEST

Xs = []
path = 'data/interim/featured_trajectories/'
for i in tqdm(range(0, 1982)): # With the parameters given there are 1982 observations
    Xs.append(np.load(path + f'{i}.npy'))
X = np.concatenate(Xs)
del Xs

Y = np.load(f'data/interim/SNPs/Ys_{my_selcted_SNP}.npy')
G = np.load('data/interim/Genotype_class/Genotype.npy')
GI = pd.read_csv('data/interim/Genotype_class/general_info.csv')
MI = pd.read_csv('./data/interim/Genotype_class/Meta_Info.csv')


### Normal splits:
os.chdir('data/processed')
subprocess.run('mkdir No_stratification', shell=True)
os.chdir(origin)
val_size = 0.2
splits = \
split_train_val_test(X, G, Y, GI, MI, stratify= False, test = True, random_state = 124,  val_size = val_size)
names =    ( 'X_train',
            'X_val',
            'X_test',
            'G_train',
            'G_val',
            'G_test',
            'y_train',
            'y_val',            
            'y_test',
            'GI_train',
            'GI_val',
            'GI_test',
            'MI_train',
            'MI_val',
            'MI_test' )
#pdb.set_trace()
for n, s in zip(names, splits):
    if n[0:2] == 'GI':
        s.to_csv(f'data/processed/No_stratification/{n}.csv')
    elif n[0:2] == 'MI':
        #print(s)
        s.to_csv(f'data/processed/No_stratification/{n}.csv')
    else:
        print(n, 'Shape: ', s.shape)
        np.save(f'data/processed/No_stratification/{n}.npy', s)

print('Normal splits done!')
del splits


### anti-Stratified split:
os.chdir('data/processed')
subprocess.run('mkdir anti-Stratified', shell=True)
os.chdir(origin)

splitter_1 = GroupShuffleSplit(test_size=.20, n_splits=1, random_state = 124)

split_1 = [split for split in splitter_1.split(X, groups = G)]

train = split_1[0][0]

test_val = split_1[0][1]

splitter_2 = GroupShuffleSplit(test_size= .50, n_splits=1, random_state = 124)

split_2 = [split for split in splitter_2.split(X[test_val], groups = G[test_val])]

val = split_2[0][0]
test = split_2[0][1]

names =  (  'X_train',
            'X_val',
            'X_test',
            'G_train',
            'G_val',
            'G_test',
            'y_train',
            'y_val',            
            'y_test',
            'GI_train',
            'GI_val',
            'GI_test',
            'MI_train',
            'MI_val',
            'MI_test' )

splits = (  X[train],
            X[test_val][val],
            X[test_val][test],
            G[train],
            G[test_val][val],
            G[test_val][test],
            Y[train],
            Y[test_val][val],
            Y[test_val][test],
            GI.iloc[train],
            GI.iloc[test_val].iloc[val],
            GI.iloc[test_val].iloc[test],
            MI.iloc[train],
            MI.iloc[test_val].iloc[val],
            MI.iloc[test_val].iloc[test] )

for n, s in zip(names, splits):
    if n[0:2] == 'GI':
        s.to_csv(f'data/processed/anti-Stratified/{n}.csv')
    elif n[0:2] == 'MI':
        s.to_csv(f'data/processed/anti-Stratified/{n}.csv')
    else:
        print(n, 'Shape: ', s.shape)
        np.save(f'data/processed/anti-Stratified/{n}.npy', s)

print('Anti-Stratified splits done!')

del splits

### Stratified split:
os.chdir('data/processed')
subprocess.run('mkdir Genotype_stratified', shell=True)
os.chdir(origin)

## REMOVING GENOTYPES WITH LESS THAN 3 REPLICATES:
index_3or_more = np.in1d(G,
np.unique(G, return_counts= True)[0][np.unique(G,
return_counts= True)[1] > 4])

## X, G, Y GETS MODIFIIED HERE
X = X[index_3or_more]
G = G[index_3or_more]
Y = Y[index_3or_more]
GI = GI[index_3or_more]
MI = MI[index_3or_more]
val_size = 0.35


splits = \
split_train_val_test(X, G, Y, GI, MI, stratify= True, test = True, random_state = 124,  val_size = val_size)
names =    ( 'X_train',
            'X_val',
            'X_test',
            'G_train',
            'G_val',
            'G_test',
            'y_train',
            'y_val',            
            'y_test',
            'GI_train',
            'GI_val',
            'GI_test',
            'MI_train',
            'MI_val',
            'MI_test' )
for n, s in zip(names, splits):
    if n[0:2] == 'GI':
        s.to_csv(f'data/processed/Genotype_stratified/{n}.csv')
    elif n[0:2] == 'MI':
        s.to_csv(f'data/processed/Genotype_stratified/{n}.csv')
    else:
        print(n, 'Shape: ', s.shape)
        np.save(f'data/processed/Genotype_stratified/{n}.npy', s)

print('Stratified splits done!')

######################################## PCA values
if my_args['pca']:
    subprocess.run('mkdir data/processed/PCA', shell=True)

    print('Not implemented.')
    exit()
    X_pc, Y_pc, genotype_pc, weights_pc, meta_info_ordered_pc = \
    generate_PCA_pairs(   Max_n,
                            Max_e,
                            Outlier_threshold,
                            X)
    #pdb.set_trace()

    weights_pc = weights_pc * (1 / np.sum(weights_pc)) # Normalize sum 1

    ## REMOVING GENOTYPES WITH LESS THAN 3 REPLICATES:
    index_3or_more = np.in1d(genotype_pc,
    np.unique(genotype_pc, return_counts= True)[0][np.unique(genotype_pc,
    return_counts= True)[1] > 4])

    if Stratify:
        X_pc = X_pc[index_3or_more]
        genotype_pc = genotype_pc[index_3or_more]
        Y_pc = Y_pc[index_3or_more]
        meta_info_ordered_pc = meta_info_ordered_pc[index_3or_more]


    X_pc_train, X_pc_val, X_pc_test, \
    G_pc_train, G_pc_val, G_pc_test, \
    y_pc_train, y_pc_val, y_pc_test, \
    GI_pc_train, GI_pc_val, GI_pc_test\
    = \
    split_train_val_test(X_pc,
                        genotype_pc,
                        Y_pc,
                        meta_info_ordered_pc,
                        stratify= Stratify, 
                        test = True, 
                        random_state = 124,  
                        val_size = val_size)

    np.save('data/processed/PCA/X_pc_train.npy', X_pc_train)
    np.save('data/processed/PCA/X_pc_val.npy', X_pc_val)
    np.save('data/processed/PCA/X_pc_test.npy', X_pc_test)
    np.save('data/processed/PCA/G_pc_train.npy', G_pc_train)
    np.save('data/processed/PCA/G_pc_val.npy', G_pc_val)
    np.save('data/processed/PCA/G_pc_test.npy', G_pc_test)
    np.save('data/processed/PCA/Y_pc_train.npy', y_pc_train)
    np.save('data/processed/PCA/Y_pc_val.npy', y_pc_val)
    np.save('data/processed/PCA/Y_pc_test.npy', y_pc_test)
    np.save('data/processed/PCA/pca_weights.npy', weights_pc)
    GI_pc_train.to_csv('data/processed/GI_pc_train.csv')
    GI_pc_val.to_csv('data/processed/GI_pc_val.csv')
    GI_pc_test.to_csv('data/processed/GI_pc_test.csv')
