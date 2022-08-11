
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from textwrap import wrap
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import os
from sklearn import preprocessing 
import sys
import rpy2.robjects as robjects
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa

def pred_performance_by_SNP(X_path, SNP, hd_name):
    '''
    Input .csv has GENOTYPE as first column. The rest are dimensions
    of the hidden representation
    '''

    ############################################ GENERATING RESPONSE
    X = pd.read_csv(X_path)

    if X['Genotype'].dtype != 'int64': # To handle the handcrafted case
        if 'col' in  np.array(X['Genotype']):
            di = np.where(np.array(X['Genotype']) == 'col')[0]
            X = X.drop(di, axis = 0)

    Genotype = np.array(X['Genotype'])

    robjects.r['load'](
        "data/raw/raw_data/R_data/LFN350acc_000_new_gene_annotation_ibd_kinship.RData")
    markers = np.array(robjects.r['GWAS.obj'][1])
    my_map = pd.read_csv("data/raw/raw_data/genotype-codes.csv")

    abrc_new = my_map['ABRC.new'].iloc[
        [list(my_map['ID.ethogenomics']).index(int(x))
        for x in np.array(Genotype) if x != 'col']] # Transform Genotype into
                                                    # the genotype format of
                                                    # marker array
    # Genotypes in marker array
    accessions = list(np.array(robjects.r['GWAS.obj'][1].colnames))

    marker = markers[0:, SNP - 1] # Select the marker

    Y = [marker[accessions.index(x)] if x in accessions else 'NA'
    for x in abrc_new] # Get the marker value in the order of X

    if X.shape[0] != (sum([y != 'NA' for y in Y])): # X includes all
        print('X contains non available markers')
        X = X.iloc[[y != 'NA' for y in Y]] # Remove the 'NA' in X
        Y = np.array(Y)[[y != 'NA' for y in Y]].astype(np.int0)
    else:
        Y = np.array(Y)
    
    assert(X.shape[0] == Y.shape[0])
    #################################################################

    ################################################# Generating pred.
    out_table = {'Dimension': [], 'Logis. Train Accuracy': [],
                    'Logis. Train MCC': [], 'Logis. Test Accuracy': [],
                    'Logis. Test MCC': [], 'Knn. Train Accuracy': [],
                    'Knn. Train MCC': [], 'Knn. Test Accuracy': [],
                    'Knn. Test MCC': [],'Prop. train': [], 
                    'Prop. test': []}

    # SPLIT:
    ## Remove genotypes with less than 3 replicates:

    index_3or_more = np.in1d(X['Genotype'],
        np.unique(X['Genotype'], return_counts= True)[0][np.unique(X['Genotype'],
        return_counts= True)[1] > 3])
    X = X.iloc[index_3or_more]
    Y = Y[index_3or_more]
    Genotype = X['Genotype']
    X = X.drop('Genotype', axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, stratify= Genotype, test_size=0.35, random_state=124)

    #First predictive performance of each dimension:
    out_table['Prop. train'] = np.mean(y_train)
    out_table['Prop. test'] = np.mean(y_test)
    

    for dim in X:
        if dim != 'Genotype':
            out_table['Dimension'].append(dim)
            ### LOGISTIC REGRESSION
            train = np.array(X_train[dim]).reshape(-1, 1)
            test = np.array(X_test[dim]).reshape(-1, 1)
            train = preprocessing.StandardScaler().fit_transform(train)
            test = preprocessing.StandardScaler().fit_transform(test)
            
            clf = LogisticRegression(random_state=0, max_iter = 500).fit(train, y_train)
            out_table['Logis. Train Accuracy'].append(clf.score(train, y_train))
            out_table['Logis. Test Accuracy'].append(clf.score(test, y_test))

            out_table['Logis. Train MCC'].append(matthews_corrcoef(clf.predict(train), y_train))
            out_table['Logis. Test MCC'].append(matthews_corrcoef(clf.predict(test), y_test))

            ### KNN
            k_range = list(np.arange(1, 30)[::3])
            param_grid = dict(n_neighbors=k_range)
            knn = KNeighborsClassifier()

            grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy',
             return_train_score=False, verbose=0)
            
            grid_search=grid.fit(train, y_train)
            
            neigh = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
            neigh.fit(train, y_train)
            
            out_table['Knn. Train Accuracy'].append(neigh.score(train, y_train))
            out_table['Knn. Test Accuracy'].append(neigh.score(test, y_test))

            out_table['Knn. Train MCC'].append(matthews_corrcoef(neigh.predict(train), y_train))
            out_table['Knn. Test MCC'].append(matthews_corrcoef(neigh.predict(test), y_test))

        else:
            print('Something weird happen!')
            exit()
    
    X_train = preprocessing.StandardScaler().fit_transform(X_train)
    X_test = preprocessing.StandardScaler().fit_transform(X_test)

    out_table['Dimension'].append('All, l1 penalty')
    clf = LogisticRegressionCV(cv=5, penalty = 'l1',
     random_state=0, solver = 'saga', max_iter = 1500).fit(X_train, y_train)
    out_table['Logis. Train Accuracy'].append(clf.score(X_train, y_train))
    out_table['Logis. Test Accuracy'].append(clf.score(X_test, y_test))

    out_table['Logis. Train MCC'].append(matthews_corrcoef(clf.predict(X_train), y_train))
    out_table['Logis. Test MCC'].append(matthews_corrcoef(clf.predict(X_test), y_test))

    out_table['Knn. Train Accuracy'].append('-')
    out_table['Knn. Test Accuracy'].append('-')

    out_table['Knn. Train MCC'].append('-')
    out_table['Knn. Test MCC'].append('-')

    # Using all of them vanilla:
    out_table['Dimension'].append('All')
    clf = LogisticRegression(random_state=0, max_iter = 1500).fit(X_train, y_train)
    out_table['Logis. Train Accuracy'].append(clf.score(X_train, y_train))
    out_table['Logis. Test Accuracy'].append(clf.score(X_test, y_test))

    out_table['Logis. Train MCC'].append(matthews_corrcoef(clf.predict(X_train), y_train))
    out_table['Logis. Test MCC'].append(matthews_corrcoef(clf.predict(X_test), y_test))

    ### KNN
    k_range = list(np.arange(1, 30)[::3])
    param_grid = dict(n_neighbors=k_range)
    knn = KNeighborsClassifier()

    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy',
        return_train_score=False, verbose=0)
    
    grid_search=grid.fit(X_train, y_train)
    
    neigh = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
    neigh.fit(X_train, y_train)
    
    out_table['Knn. Train Accuracy'].append(neigh.score(X_train, y_train))
    out_table['Knn. Test Accuracy'].append(neigh.score(X_test, y_test))

    out_table['Knn. Train MCC'].append(matthews_corrcoef(neigh.predict(X_train), y_train))
    out_table['Knn. Test MCC'].append(matthews_corrcoef(neigh.predict(X_test), y_test))

    output_df = pd.DataFrame.from_dict(out_table)
    print(output_df)
    output_df.to_csv(f'results/pred_performance_{hd_name}.csv')

def sweep_whole_genome(X_path, hd_name, col_name =  None):
    ############################################ GENERATING RESPONSE
    X = pd.read_csv(X_path)

    if X['Genotype'].dtype != 'int64': # To handle the handcrafted case
        if 'col' in  np.array(X['Genotype']):
            di = np.where(np.array(X['Genotype']) == 'col')[0]
            X = X.drop(di, axis = 0)

    Genotype = np.array(X['Genotype'])

    robjects.r['load'](
        "data/raw/raw_data/R_data/LFN350acc_000_new_gene_annotation_ibd_kinship.RData")

    markers = np.array(robjects.r['GWAS.obj'][1])
    my_map = pd.read_csv("data/raw/raw_data/genotype-codes.csv")

    abrc_new = my_map['ABRC.new'].iloc[
        [list(my_map['ID.ethogenomics']).index(int(x))
        for x in np.array(Genotype) if x != 'col']] # Transform Genotype into
                                                    # the genotype format of
                                                    # marker array
    # Genotypes in marker array
    accessions = list(np.array(robjects.r['GWAS.obj'][1].colnames))


    Y_index = [accessions.index(x) if x in accessions else 'NA'
    for x in abrc_new] # Get the marker value in the order of X
    my_markers = [np.expand_dims(markers[accessions.index(x), :].astype(np.int8), 0) if x in accessions else 'NA'
    for x in abrc_new]

    if X.shape[0] != (sum([y != 'NA' for y in Y_index])): # X includes all
        print('X contains non available markers')
        X = X.iloc[[y != 'NA' for y in Y_index]] # Remove the 'NA' in X
        Y = np.array(my_markers, dtype=object)[[y != 'NA' for y in Y_index]]
    else:
        Y = my_markers
    
    Y = np.concatenate(Y)
    print(Y.shape)
    assert(X.shape[0] == Y.shape[0])
    # SPLIT:
    ## Remove genotypes with less than 3 replicates:

    index_3or_more = np.in1d(X['Genotype'],
        np.unique(X['Genotype'], return_counts= True)[0][np.unique(X['Genotype'],
        return_counts= True)[1] > 3])
    X = X.iloc[index_3or_more]
    Y_index = np.arange(index_3or_more.shape[0])[index_3or_more]
    Genotype = X['Genotype']
    X = X.drop('Genotype', axis = 1)

    if col_name == None:
        pass
    else:
        X = np.array(X[col_name]).reshape(-1, 1)

    X_train, X_test, y_train_index, y_test_index = train_test_split(
        X, Y_index, stratify= Genotype, test_size=0.35, random_state=124)

    out_table = {'SNP':[], 'Train Accuracy': [],
                'Train MCC': [], 'Test Accuracy': [],
                'Test MCC': [], 'Prop. train': [], 
                'Prop. test': []}

    problematic_values = []

    X_train = preprocessing.StandardScaler().fit_transform(X_train)
    X_test = preprocessing.StandardScaler().fit_transform(X_test)

    for i in tqdm(range(Y.shape[1])):
        
        y_train = Y[y_train_index, i]
        y_test = Y[y_test_index, i]
        out_table['Prop. train'].append(np.mean(y_train))
        out_table['Prop. test'].append(np.mean(y_test))
        out_table['SNP'].append(i + 1)
        
        clf = svm.SVC().fit(X_train, y_train)
        aveg_test = np.mean(y_test)
        if aveg_test == 1 or aveg_test == 0:
            problematic_values.append(i)
            continue
        
        out_table['Train Accuracy'].append(clf.score(X_train, y_train))
        out_table['Test Accuracy'].append(clf.score(X_test, y_test))

        out_table['Train MCC'].append(matthews_corrcoef(clf.predict(X_train), y_train))
        out_table['Test MCC'].append(matthews_corrcoef(clf.predict(X_test), y_test))
        
    plt.scatter(out_table['SNP'], out_table['Test MCC'])
    #plt.show()
    output_df = pd.DataFrame.from_dict(out_table)
    
    output_df.to_csv(f'results/{hd_name}_reverseGwas_214k.csv', index=False)


def calculate_correlations(X_path, Stratified: bool, hd_name):
    ## _ // _ // Trial // Arena // Genotype // Plant // Export file
    if not Stratified:
        MI_train = pd.read_csv('data/processed/No_stratification/MI_train.csv', index_col=False)
        MI_val = pd.read_csv('data/processed/No_stratification/MI_val.csv', index_col=False)
        MI_test = pd.read_csv('data/processed/No_stratification/MI_test.csv', index_col=False)
        MI = np.vstack((MI_train, MI_val, MI_test))

        del MI_train, MI_val, MI_test
    else:
        MI_train = pd.read_csv('data/processed/Genotype_stratified/MI_train.csv', index_col=False)
        MI_val = pd.read_csv('data/processed/Genotype_stratified/MI_val.csv', index_col=False)
        MI_test = pd.read_csv('data/processed/Genotype_stratified/MI_test.csv', index_col=False)
        MI = np.vstack((MI_train, MI_val, MI_test))

        del MI_train, MI_val, MI_test

    MI = pd.DataFrame(MI)
    MI.columns = ['Leaf', 'AnalysisGroup',  'Trial', 'Arena', 'Genotype', 'Plant', 'Export file']
    MI = MI.drop(['Leaf', 'Export file'], 1)

    
    #print(MI.shape)
    #### TMP TODO generate metainfo in order
    #MI.to_csv(f'data/processed/Hidden_representations/supervised/{hd_name}_metainfo.csv', index = False)
    #exit()
    ####
    X_AI = pd.read_csv(X_path)
    X_H = pd.read_csv('data/processed/Hidden_representations/Hand_features/handcrafted_dimensions.csv')
    #X_H_metainfo = pd.read_csv('data/processed/Hidden_representations/Hand_features/handcrafted_dimensions_meta_info.csv').drop(['Leaf', 'AnalysisGroup'], 1)
    X_H_metainfo = pd.read_csv('data/processed/Hidden_representations/Hand_features/handcrafted_dimensions_meta_info.csv').drop(['Leaf'], 1)
    #print(X_H_metainfo.shape)



    MI_list = list( MI['Genotype'].astype(str) +  MI['Plant'].astype(str)) 

    X_H_metainfo_list = list(
        X_H_metainfo['Genotype'].astype(str) +
        X_H_metainfo['Plant'].astype(str))
    
   
    X_AI_index  = [MI_list.index(x) if x in MI_list else -1 for x in X_H_metainfo_list]

    my_bool = np.array(X_AI_index) != -1
    My_index = np.array(X_AI_index)[np.where(my_bool)[0]]
    #print(np.array(X_H_metainfo_list)[my_bool])
    #print(np.array(MI_list)[My_index])

    X_AI = X_AI.iloc[My_index, :]
    X_H = X_H.iloc[my_bool, :]

    #rint(X_AI.shape)
    #print(X_H.shape)
    
    scaler = StandardScaler()
    
    
    X_AI_s = scaler.fit_transform(X= X_AI.drop('Genotype', 1))
    X_H_s = scaler.fit_transform(X = X_H.drop('Genotype', 1))
    pca = PCA(n_components=1)
    cca = CCA(n_components=1)
    # PCA
    pca_AI = pca.fit_transform(X_AI_s)
    pca_H = pca.fit_transform(X_H_s)
    
    #print(pca_AI)
    #print(pca_H)

    print('PCA:', np.corrcoef(np.ravel(pca_AI), np.ravel(pca_H))[0, 1])

    #CCA

    cca.fit(X_AI_s, X_H_s)
    X_c, Y_c = cca.transform(X_AI_s, X_H_s)
    #print(X_c.shape)
    #print(Y_c.shape)
    

    print('CCA:', np.corrcoef(np.ravel(X_c), np.ravel(Y_c))[0, 1])

    #plt.scatter(pca_AI, pca_H, s = 0.5, c = 'black')
    #plt.xlabel('Generated traits PC1')
    #plt.ylabel('Handcrafted traits PC1')
    #plt.title('PCA correlation')
    ##plt.show()
#
    #plt.scatter(X_c, Y_c, s = 0.5, c = 'black')
    #plt.xlabel('Generated traits CCA1')
    #plt.ylabel('Handcrafted traits CCA1')
    #plt.title('CCA correlation')
    ##plt.show()

    ## Extract probabilities
    

    my_model = tf.keras.models.load_model('models/trained_models/Supervised/IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS0_snp86001_BS20_0.h5',
    custom_objects = {'Addons>MatthewsCorrelationCoefficient': tfa.metrics.MatthewsCorrelationCoefficient(num_classes= 2)})
    
    print('Loading stratified splits...')
    X_train = np.load('data/processed/Genotype_stratified/X_train.npy')[:, ::1, np.arange(0, 21)]
    X_val = np.load('data/processed/Genotype_stratified/X_val.npy')[:, ::1, np.arange(0, 21)]
    X_test = np.load('data/processed/Genotype_stratified/X_test.npy')[:, ::1, np.arange(0, 21)]
    print('X shape', X_train.shape)

    Big_X = np.vstack((X_train, X_val, X_test))
    my_probabilities = my_model.predict(Big_X, 5)
    print(my_probabilities)
    print(my_probabilities.shape)
    
    my_probabilities_0 = my_probabilities[My_index, 1]
    my_probabilities_1 = my_probabilities[My_index, 0]
    prob_corr_0 = []
    prob_corr_1 = []
    names = []
    print('Probabilities corr...')
    for i, col in enumerate(X_H.columns):
        if col == 'Genotype':
            continue
        names.append(col)
        prob_corr_0.append(np.corrcoef(np.ravel(my_probabilities_0), np.ravel(X_H[col]))[0, 1])
        prob_corr_1.append(np.corrcoef(np.ravel(my_probabilities_1), np.ravel(X_H[col]))[0, 1])
        #print(np.corrcoef(np.ravel(my_probabilities), np.ravel(X_H[col]))[0, 1])
    
    #names = [ '\n'.join(wrap(l, 20)) for l in names ]

    figure(figsize=(15, 120), dpi= 60)
    my_colors = ['red' if x < 0 else 'green' for x in prob_corr_1]

    plt.barh(names, prob_corr_1, linewidth = 1, edgecolor = 'black',
                color = my_colors)
    plt.tick_params('y', width= 4)
    plt.tick_params('x', width= 0.5)
    plt.title('Correlations with probability of marker score 1\nSNP 86001', fontsize = 25)
    plt.savefig(f'results/{hd_name}_Correlations_with_probability_of_1.png')
    plt.yticks(fontsize = 10)
    plt.xlabel('Pearson correlation')
    #plt.xticks(rotation = 90, fontsize = 30)
    #plt.show()
    #figure(figsize=(10, 8), dpi= 50)
    #plt.tight_layout()
    #plt.barh(names, prob_corr_1)
    #plt.title('Correlations with probability of 1')
    #plt.savefig(f'results/{hd_name}_Correlations_with_probability_of_1.png')
    #plt.yticks(fontsize = 15)
    ##plt.xticks(rotation = 90, fontsize = 30)
    #plt.show()

#    figure(figsize=(20, 10), dpi=80)
#    plt.barh(names, prob_corr_0)
#    plt.title('Correlations with probability of 0')
#    plt.savefig(f'results/{hd_name}_Correlations_with_probability_of_0.png')
#    plt.yticks(fontsize = 30)
#    #plt.xticks(rotation = 90, fontsize = 30)
#    plt.show()
#
    correlations = np.array(X_AI.drop('Genotype', 1).columns).reshape(len(X_AI.drop('Genotype', 1).columns), 1)
    
    for i, col in enumerate(X_H.columns):
        if col == 'Genotype':
            continue
        #print(np.array(X_AI.drop('Genotype', 1).corrwith(X_H[col])))
        tmp_cor = np.expand_dims(np.array(X_AI.drop('Genotype', 1).corrwith(X_H[col])), axis=1)
        correlations = np.hstack([correlations, tmp_cor])
        #print(X_AI.drop('Genotype', 1).corrwith(X_H[col]))
    correlations = pd.DataFrame(correlations[:, 1:], index= correlations[:, 0]).astype(float)
    correlations.columns = list(X_H.drop('Genotype', 1).columns)
    #correlations.index  = (X_AI.drop('Genotype', 1).index)
    print(correlations)
    correlations.to_csv(f'results/{hd_name}_correlations.csv')
    figure(figsize=(60, 20), dpi=400)
    ax = sns.heatmap(correlations.T, annot=False, cmap = 'jet',
    cbar_kws = dict(use_gridspec=False, location="top", shrink = 0.3), linewidths=0.8, linecolor="grey")
    
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)
    plt.yticks(fontsize = 30)
    plt.xticks(rotation = 90, fontsize = 30)
    plt.savefig(f'results/{hd_name}_correlations.png')
    


# Correlations EthoAnalysis and Supervised
calculate_correlations('data/processed/Hidden_representations/supervised/IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_snp86001_BS20.csv',
 1,
'/Supervised/IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_snp86001_BS20')

# REVERSE GWAS RUN:
# Hancrafted:
sweep_whole_genome("data/processed/Hidden_representations/Hand_features/handcrafted_dimensions.csv",
                    'Hand_features/Handcrafted_dimensions')
# Supervised:
sweep_whole_genome("data/processed/Hidden_representations/supervised/IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_snp86001_BS20.csv",
                    'Supervised/IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_snp86001_BS20')
# Self-Supervised:
sweep_whole_genome("data/processed/Hidden_representations/self-supervised/MINIMAL_VAE_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_BS32.csv",
                    'Self-Supervised/MINIMAL_VAE_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_BS32')
# Contrastive:
sweep_whole_genome("data/processed/Hidden_representations/Contrastive/MINIMAL_Contrastive_C1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_BS128.csv",
                    'Contrastive/MINIMAL_Contrastive_C1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_BS128')
    

