import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
import os
import sys
import matplotlib.pyplot as plt

## add directory to import modules to the list
WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../" + "/../"
sys.path.insert(1, WORKDIR_PATH)

# Import relevant modules
from src.features.process_data import add_features, normalize_temporal_series

# undo changes
WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, WORKDIR_PATH)


class data_handler():

    '''
    A class to make it easier handling the data loading.
    It needs the path to all relevant folders.

    -> max_n = number of consecutive time steps allowed to have missing values. They will imputed.
    -> max_e = number values to remove at the end of ALL series so series with missing values at
    the beginning can be included.

    ----------------------------------------------------------------------------------------
    Usage:
    First a pass of make XY pairs is needed to load all required data. After that, it is possible to
    output just the SNP values with just_Y in a faster way.
    '''
    def __init__(self,  max_n = 150, max_e = 500, outlier_trheshold = 20, 
                    trajectories_folder = "data/raw/raw_data/Trajectories_by_arena/",
                    arena_info_path = "data/raw/raw_data/_arenaInfos.csv",
                    marker_data_path = "data/raw/raw_data/R_data/LFN350acc_000_new_gene_annotation_ibd_kinship.RData",
                    genotype_map_path = "data/raw/raw_data/genotype-codes.csv"):

        self.trajectories_folder = trajectories_folder
        self.arena_info_path = arena_info_path
        self.marker_data_path = marker_data_path
        self.genotype_map_path = genotype_map_path
        self.max_n = max_n
        self.max_e = max_e
        self.ot = outlier_trheshold # 2mm per second

    def _data_loader(self, na_report = True, give_me_plant_info = False):
        '''
        Loads data into a list of np.arrays. If na_report = False it only returns a list with the data,
        otherwise it outputs a tuple of lists with iformation regarding NaN.

        -> Trajectories_folder: Path to the foler where all the trajectories are stored.
        The trajectories consist of .csv files with up to 25502 rows of cartesian coordinates, each
        trajectory corresponds to a unique arena.

        -> arena_info_path: Path to where the information about each arena is stored. This contains
        information of the genotype of the plant and the .csv document name inside the trajectories folder. It
        allows to map each trajectory with its genetic information.

        Returns a tuple of 4 lists (or just the first two):
        -> X: List of trajectories as a np.array with shape 25498x2.
        -> Genotype_info: Genotype of the plant in the arena.
        -> nan_register: # of NaN.
        -> na_position: Position of NaN in the data.

        '''
        Arena_infos = pd.read_csv(self.arena_info_path)#.iloc[1:100, :] #debugging
        # Remove problematic observation with weird genotype name:
        Arena_infos = Arena_infos.loc[Arena_infos['Genotype'] != 'col']  # I Have changed this in excel file!

        nan_register = []
        na_position = []
        genotype_info = [] # Genotype value per arena.
        plant_info = []
        X = []
       
        print('Loading data...')
        if na_report:
            for i in tqdm(range(Arena_infos.shape[0])): # Reads arenas in order
                # 25498 is to remove values outside minimum range of available data and unify the shape.
                x = np.array(pd.read_csv(self.trajectories_folder + Arena_infos.iloc[i]['Export file']))[:25498]
                genotype_info.append(Arena_infos.iloc[i]['Genotype'])
                nan_register.append(sum(np.isnan(x)))
                na_position.append(np.where(np.isnan(x)))
                X.append(x)
            print('Data loaded!')
            self.original_data = X
            return(X, genotype_info, nan_register, na_position)
        else:
            for i in tqdm(range(Arena_infos.shape[0])): # Reads arenas in order
                # 25498 is to remove values outside minimum range of available data and unify the shape.
                x = np.array(pd.read_csv(self.trajectories_folder + Arena_infos.iloc[i]['Export file']))[:25498]
                genotype_info.append(Arena_infos.iloc[i]['Genotype'])
                plant_info.append(Arena_infos.iloc[i]['Plant'])
                X.append(x)
            print('Data loaded!')
            self.original_data = X
            if give_me_plant_info:
                return(X, genotype_info, plant_info)
            else: 
                return(X, genotype_info)
    
    def _impute_data(self, X: list): 
        '''
        This function does two things:
            1- If there are up to max_n points of missing data between two available points, it distributes 
            the points evenly in the straight line between them.
            2- If there is missing data at the beginning, it copies the closest available point. 
            In the worst case scenario we could have 150 time steps in the same position in this case.
            
        ------
        max_n = Number of maximum time steps above or below to find data available.

        Returns a tuple with a list of trajectories or None values and a list of
        the available arenas (the positions in the previous list). If None it means
        that the arena in that position have not been included due to missing data.
        '''
        def b_point(X, i, j, n = 0):
            '''
            Look recursively for the closest 'before' point until max_n.
            '''
            pos = j - 1
            if j - 1 < 0:
                return None, None # It is not possible to have missing values in the first entry if there is any non missing value

            point = X[i][j-1, 1:]

            if n >= self.max_n:
                return None, None
            if np.isnan(point[0]):
                point, pos = b_point(X, i, j-1, n = n + 1) # Recurssion here adds nothing!
            return(point, pos)

        def a_point(X, i, j, n = 0):
            '''
            Look recursively for the closest 'after' point until max_n.
            If the step is after 25498, inputed just the inmedate value before.
            '''
            pos = j + 1
            if j + 1 >= (25498 - self.max_e): 
                return None, None
               

            point = X[i][j+1, 1:]
            if n >= self.max_n:
                return None, None
            if np.isnan(point[0]):
                point, pos = a_point(X, i, j+1, n = n + 1)
            return(point, pos)
        
        def remove_initial_nan(X: np.array) -> np.array:
            '''
            Move the series to the left. Removing initial missing values and
            adding them to the end. 

            It is considered to be a non NAN if there are at least 5 consecutive values
            with a non NAN.
            '''
            first_non_nan = 0
            #consecutive_non_nan = 0
            for i, x in enumerate(X):
                if not any([np.isnan(j[1]) for j in X[i:i+5]]):
                #if not np.isnan(x[1]):
                    first_non_nan = i
                    break
            #Placeholder with missing values to accomodate reduced X series
            new_X = np.repeat(np.array([[np.NaN, np.NaN, np.NaN]]), 25498 , 0)
            # Input it
            new_X[:(25498-first_non_nan), :] = X[first_non_nan:, :]
            assert not (np.isnan(new_X[0, 0]))
            return new_X


        too_many_missing = 0
        print('Preparing data...')
        for i in tqdm(range(len(X))):
            X[i] = remove_initial_nan(X[i]) # Move values to the right when NaN at the beginning
            X[i] = X[i][:(25498 - self.max_e), :] # Remove values beyond the limits. Get all series smaller by max_e.
            # Impute the average between closests points:
            for j in range(X[i].shape[0]):

                # Calculate speed of the point:
                if j > 0:
                    distance = np.linalg.norm(X[i][j, 1:] - X[i][j - 1, 1:])
                    if np.abs(distance / 0.2) >= self.ot: # Check if outlier
                        X[i][j, 1:] = np.nan # If outlier, add nan

                if np.isnan(X[i][j, 1]): # check if nan
                    previous, p_pos = b_point(X, i, j) # get previous value and relative position
                    next, a_pos = a_point(X, i, j) # get next value and relative position
                    if (previous is None or next is None):
                        too_many_missing += 1
                        break # Not interested in this arena anymore.
                    assert a_pos-p_pos >= 0 # sanity check
                    if a_pos-p_pos == 0: # Basically the first one is missing.
                        X[i][j, 1:] = next # Just add the next one since we have no reference.

                    X[i][p_pos:a_pos + 1, 1] = np.linspace(previous[0], next[0], num = a_pos-p_pos + 1) #Impute data evenly distributed in a straight line
                    X[i][p_pos:a_pos + 1, 2] = np.linspace(previous[1], next[1], num = a_pos-p_pos + 1)

                    if j > 0: 
                        distance = np.linalg.norm(X[i][j, 1:] - X[i][j - 1, 1:])
                        if np.abs(distance / 0.2) >= 20: # This really should not happen. And it does not happen. 
                            # However, it is here to deal with a situation where the imputation generates more outliers.
                            print('Still and outlier!')
                            print(X[i][j, 1:], X[i][j-1, 1:], X[i][j + 1, 1:])
                            w = 1
                            while np.abs(distance / 0.2) >= 20:
                                distance = np.linalg.norm(X[i][j, 1:] - X[i][j - 1, 1:])
                                w += 1
                                assert p_pos-w >= 0
                                X[i][p_pos:a_pos, 1] = np.median(X[i][p_pos-w:a_pos+w, 1])
                                X[i][p_pos:a_pos, 2] = np.median(X[i][p_pos-w:a_pos+w, 1])
                                print('Window is: ', w)

                    assert not (np.isnan(X[i][:j, :])).any() # Sanity check again
        self.clean_X = []

        # Remove arenas with missing data but mantain list structure to map with gene data.
        # So, I add None when an arena with STILL missing data exists.
        print('===============================================================================================')
        print('There are %s' %too_many_missing, 'arenas with more than %s' %self.max_n, 'consecutive missing time-steps')
        available_arenas = []
        for i in range(len(X)):
            if (np.isnan(X[i])).any():
                self.clean_X.append(None)
            else:
                self.clean_X.append(X[i])
                available_arenas.append(i)
        self.available_arenas = available_arenas
        return self.clean_X, available_arenas
    
    def _map_SNP(self, marker_pos: int, trajectory_data: list, genotype_info: list, 
        available_arenas: list, load_X = True):
        '''
        Map arenas with its SNP value for a given marker.
        -> marker_pos: Position of the marker in the 214k marker array e.g. 86001.
        -> trajectory_data: The list of trajectories.
        -> genotype_info: The list of genotype IDs
        -> available_arenas: List of the position of available arenas after imputation
        -> marker_data_path: RData path
        -> genotype_map:  genotype-codes.csv path

        Returns:
        A tuple with X Y like np.arrays. Each trajectory is mapped to a SNP
        value given a marker.
        '''
        robjects.r['load'](self.marker_data_path) #Load R object

        # Matrix with the binary marker data 350x214051:
        markers = np.array(robjects.r['GWAS.obj'][1]) #This gets transposed when making a np.array
        

        # Names for each genetic variant. They correspond to the row names in markers data
        accessions = np.array(robjects.r['GWAS.obj'][1].colnames)

        # Indexing the genotype info based on available data given an imputation criteria:
        genotype_class = np.array(genotype_info)[available_arenas] 

        Arena_infos = pd.read_csv(self.arena_info_path)
        Arena_infos = Arena_infos.loc[Arena_infos['Genotype'] != 'col']
        metainfo = Arena_infos.iloc[available_arenas]

        # genotype-codes.csv. Maps the rowname of the markers object to the ID of the genotype
        # class.
        my_map = np.array(pd.read_csv(self.genotype_map_path))[:, 1:3]

        marker = markers[0:, marker_pos - 1] # Get binary 1D corresponding to given marker.
       
        J = 0 # to count non-available data
        SNP_values = []
        Trajectories = []
        G = []
        MI = []
        # Get trajectory data without None
        trajectory_data = [trajectory_data[i] for i in available_arenas] 

        for k, i in enumerate(genotype_class): # For each element in the list of genetic IDs
            genetic_info = my_map[np.where(my_map[:, 1] == int(i)), 0][-1] # Get row name1
            _SNP_value =  marker[np.where(accessions == genetic_info)] # Get value for that row
            if _SNP_value.size == 0:
                print('genotype, ', genetic_info, 'has no SNP information')
                J += 1
                continue # This arena has no genetic information
            SNP_values.append(_SNP_value[0])
            if load_X:
                Trajectories.append(trajectory_data[k].reshape(1, trajectory_data[k].shape[0], trajectory_data[k].shape[1]))
                G.append(i)
                MI.append(metainfo.iloc[k])
        print('===============================================================================================')
        print('There are', J, 'arenas with no genetic information. There are a total of', len(SNP_values),
         'arenas available for marker', marker_pos)
        print('===============================================================================================')
        percentaje_of_ones = (sum(SNP_values) / len(SNP_values))
        print('%s of the arenas have SNP with value = 1' %percentaje_of_ones)
        if load_X:
            return np.concatenate(Trajectories, 0), np.array(G), np.array(MI), np.array(SNP_values)
        else:  
            return np.array(SNP_values)

    def makeXYpairs(self, marker: list):
        '''
        Returns a clean, ready to go pair of np.arrays given a marker value.
        '''
        print('Making pairs for marker %s ...' %marker)
        X, self.genotype_info, nan_register, na_position = self._data_loader(na_report= True)

        self.X, available_arenas = self._impute_data(X)


        X, G, MI,  Y = self._map_SNP(marker_pos = marker, trajectory_data = self.X,
        genotype_info = self.genotype_info, available_arenas = available_arenas)



        return X, G, pd.DataFrame(MI), np.array([[y, 1- y] for y in Y]), nan_register, na_position

    def just_Y(self, marker: int):
        '''
        Returns an array for the Y values given a marker. Must be used after making the firs XY pair.
        '''

        Y = self._map_SNP(marker_pos = marker, trajectory_data = self.X,
        genotype_info = self.genotype_info, available_arenas = self.available_arenas, load_X = False)

        return(np.array([[y, 1- y] for y in Y]))

    def genotype(self):
        return  np.array(self.genotype_info)[self.available_arenas] 

    def map_components(self, algorithm = 'PCA', markers_path = 'data/raw/raw_data/R_data/hapmap_imputed_snps_1M.RData', components = 50):
        '''
        Returns ordered component values, ready to use with the trajectory data.
        '''
        # Would be much easier to just work with a output list from X but anyways...
        robjects.r['load'](markers_path)
        markers = np.array(robjects.r['gData1M'][1]).astype(np.float16)
        rownames = np.array(robjects.r['gData1M'][1].rownames, dtype= np.int0)


        robjects.r['load'](self.marker_data_path) # Load the other R object to remove some unavailable observations:
        accessions = np.array(robjects.r['GWAS.obj'][1].colnames)
        markers_214 = np.array(robjects.r['GWAS.obj'][1]) #This gets transposed when making a np.array


        X, self.genotype_info = self._data_loader(na_report= False)
        self.X, available_arenas = self._impute_data(X)
        
        gen_map = np.array(pd.read_csv(self.genotype_map_path))

        genotype_class = np.array(self.genotype_info)[available_arenas] 

        if algorithm == 'PCA':
            pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(components))])
            red_mark = pipeline.fit_transform(markers)
            weights = pipeline['pca'].explained_variance_ratio_
        if algorithm == 't-SNE':
            pipeline = Pipeline([('scaling', StandardScaler()), ('TSN', TSNE(2))])
            red_mark = pipeline.fit_transform(markers)
           

        component_values = []
        X_with_data = []
        genotype_indicator = []
        c = -1
        for k, i in tqdm(enumerate(genotype_class)): # The key here is that these are in the same order!
            genetic_info = gen_map[np.where(gen_map[:, 2] == int(i)), 1][-1] # Get row name1
            _SNP_value =  markers_214[np.where(accessions == genetic_info)] # Get value for that row
            if _SNP_value.size == 0:
                print('Not available in 214k')
                continue
            c +=1

            MagnusID = gen_map[np.where(gen_map[:, 2] == int(i)), 6][-1]# Get row name
            try: 
                MagnusID = int(MagnusID)
            except:
                print('Not available in 1M')
                continue
            values = np.array(red_mark[np.where(rownames == MagnusID)[0], :])
            if values.size == 0:
                print('Not available in 1M')
                continue
            component_values.append(values)
            X_with_data.append(c)
            genotype_indicator.append(i)
        Y = np.concatenate(component_values, 0)
       

        assert(len(X_with_data) == Y.shape[0])
        #np.save(f'./data/interim/transformed_response/1M_SNP/{algorithm}_Ys.npy', Y)
        #np.save(f'./data/interim/transformed_response/1M_SNP/X_index_1M.npy', X_with_data)
        if algorithm == 'PCA':
            #np.save(f'./data/interim/transformed_response/1M_SNP/PCA_Weights.npy', np.array(weights))
        #np.save(f'./data/interim/transformed_response/1M_SNP/GenoType_class_1M.npy', np.array(genotype_indicator))
            return Y, X_with_data, np.array(weights), np.array(genotype_indicator)

    def ethodata(self, ethodata_path = 'data/interim/Hand_features/Pheno_2017_clean.csv'):
        '''
        Returns X and Y. However, Y is a matrix, each column correspond to each SNP. In proper
        order wrt X.
        '''
        robjects.r['load'](self.marker_data_path)
        X_pandas = pd.read_csv(ethodata_path)
        X = np.array(X_pandas.loc[:, X_pandas.notna().all()])
        #print(X_pandas.loc[:, X_pandas.notna().all()].columns[6:])
        my_map = np.array(pd.read_csv(self.genotype_map_path))[:, 1:3]
        markers = np.array(robjects.r['GWAS.obj'][1])
        accessions = np.array(robjects.r['GWAS.obj'][1].colnames) 
        Y = []
        clean_X = []
        #print('Markers matix shape: ', markers.shape)
        #print('Available genotypes in marker matrix: ', accessions)
        #print(accessions.shape)
        for x in X:
            try:
                int(x[2])
            except:
                 continue
            row_name = my_map[np.where(my_map[:, 1] == int(x[2])), 0][-1]
            SNP_values =  markers[np.where(accessions == row_name)]
           
            if SNP_values.size == 0:
                #print('genotype, ', row_name, 'has no SNP information')
                continue
            
            Y.append(SNP_values)
            clean_X.append(x[6:].reshape(1, x[6:].shape[0]))
        
        #print('X shape: ', np.concatenate(np.array(clean_X), 0).shape)
        return np.concatenate(np.array(clean_X), 0), np.concatenate(np.array(Y), 0), X_pandas.loc[:, X_pandas.notna().all()].columns[6:]

def generate_PCA_pairs(Max_n, Max_e, Outlier_threshold, X):
    
    my_data_handler = data_handler( max_n = Max_n,
                                    max_e = Max_e,
                                    outlier_trheshold = Outlier_threshold)
    
    pc_values, X_index, weights, genotype = my_data_handler.map_components(
        components = 50)

    new_X = X[X_index]

    meta_info = pd.read_csv('data/raw/raw_data/genotype-codes.csv')
    meta_info_ordered_pc  = \
    pd.concat(
        [
            meta_info.iloc[
                np.where(meta_info['ID.ethogenomics'] == x)
                ] for x in genotype])

    

    return new_X, pc_values, genotype, weights, meta_info_ordered_pc


def create_dataset(snps: list, Max_n, Max_e, Outlier_threshold, Downsampling_factor, Derivatives,
Spe_add, Ang_speed, Trav_dist, Str_line, Acc_add):
    '''
    Creates a the dataset from scratch. Cleans the data, imputes, downsamples and adds features.
    Outputs to interim.
    Returns nan report.
    '''
    my_data_handler = data_handler(max_n = Max_n, max_e = Max_e, outlier_trheshold = Outlier_threshold)
    SNP = snps
    directory = './data/interim/featured_trajectories'
    selected_Ys = dict()
 

    X, G, arena_metainfo, y_tmp_1, nan_register, na_position = my_data_handler.makeXYpairs(SNP[0])

    arena_metainfo.to_csv('./data/interim/Genotype_class/Meta_Info.csv')
    #exit() #tmp
    np.save('./data/interim/Genotype_class/' + 'Genotype.npy', G)
    selected_Ys[SNP[0]] = (y_tmp_1)

    for snp in range(len(SNP) - 1):
        selected_Ys[SNP[snp + 1]] = my_data_handler.just_Y(SNP[snp+1])

    for i, x in enumerate(X):
        print('Adding features (speed, angular speed...), this may take a while...')
        full_X = add_features(x, downsampling_factor= Downsampling_factor,
         derivatives = Derivatives, spe_add = Spe_add, acc_add = Acc_add, 
         ang_speed = Ang_speed, trav_dist = Trav_dist, str_line = Str_line) 
        full_X = normalize_temporal_series(full_X).astype(np.float32)
        np.save(directory + f'/{i}.npy', full_X)
        if i % 300 == 0:
            print(f'Arena {i}...')


    for _snp in selected_Ys:
        np.save(
            './data/interim/SNPs' + 
            '/Ys_' + str(_snp) +
             '.npy', selected_Ys[_snp])

        print('Saved ', _snp, selected_Ys[_snp].shape)


    print('Done!, X shape is: ', full_X.shape)
    np.save('./data/interim/missing_data_info/nan_position.npy', na_position)
    np.save('./data/interim/missing_data_info/nan_register.npy', nan_register)
    
    ## Add a file with genotype-codes info IN ORDER
    print('Creating -General info- df.')
    meta_info = pd.read_csv('data/raw/raw_data/genotype-codes.csv')
    meta_info_ordered  = \
    pd.concat(
        [
            meta_info.iloc[
                np.where(meta_info['ID.ethogenomics'] == x)
                ] for x in G])

    meta_info_ordered.to_csv('./data/interim/Genotype_class/general_info.csv')

