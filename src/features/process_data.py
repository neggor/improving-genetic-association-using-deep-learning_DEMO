from xmlrpc.client import Boolean
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
from sklearn.model_selection import train_test_split


def add_speed_acceleration(trajectory: np.array) -> np.array:
    '''
    Returns speed, acceleration.

    It approximates the distance as a straight line 
    between the two points at the two different time steps.
    '''
    time_steps = trajectory.shape[0]
    speed = [0]
    acceleration = [0]

    for i in range(1, time_steps):
        # Speed:
        distance = np.linalg.norm(trajectory[i, 1:] - trajectory[i - 1, 1:])  #L2 norm (Euclidean distance))
        time = trajectory[i, 0] - trajectory[i - 1, 0]
        speed.append(np.abs(distance / time))
        # Acceleration:
        diff_vel = speed[i] - speed[i - 1]
        acceleration.append(diff_vel / time)

    speed = np.array(speed).reshape([time_steps, 1])
    acceleration = np.array(acceleration).reshape([time_steps, 1])

    return speed, acceleration

def add_angular_speed(trajectory: np.array) -> np.array:
    '''
    Returns relative (w.r.t previous point) angular speed
    '''
    time_steps = trajectory.shape[0]
    angular_speed = [0]
    for i in range(1, time_steps):
        time = trajectory[i, 0] - trajectory[i - 1, 0]
        x1 = np.array(trajectory[i, 1:], dtype= 'float')
        x2 = np.array(trajectory[i - 1, 1:], dtype= 'float')

        Lx1 = np.linalg.norm(x1)
        Lx2 = np.linalg.norm(x2)

        cross = np.cross(x1, x2)

        if Lx1 * Lx2 == 0:
            angle_abs = 0
        else:
            if cross > 0:
                sign = 1
                cos_ang = np.clip(x1.dot(x2) / (Lx1 * Lx2), -1, 1)
                angle_abs = np.arccos(cos_ang)
            elif cross < 0:
                sign = -1
                cos_ang = np.clip(x1.dot(x2) / (Lx1 * Lx2), -1, 1)
                angle_abs = np.arccos(cos_ang)
            else: # Case cross-product is 0
                sign = 1
                if x1[0] != 0:
                    if (x1[0] * x2[0] > 0.0):
                        angle_abs = 0 # same direction
                    else:
                        angle_abs = -np.pi # opposite direction
                else:
                    if (x1[1] * x2[1] > 0.0):
                        angle_abs = 0
                    else:
                        angle_abs = -np.pi
    
        angular_speed.append((sign * angle_abs)/time)
    return  np.array(angular_speed).reshape(time_steps, 1)

def travel_distance(trajectory):
    time_steps = trajectory.shape[0]
    distance = [0]
    for i in range(1, time_steps):
        distance.append(np.linalg.norm(trajectory[i, 1:] - 
        trajectory[i - 1, 1:]) + distance[i-1]) 

    return np.array(distance).reshape(time_steps, 1)

def straigth_line(trajectory):
    time_steps = trajectory.shape[0]
    distance = [0]
    for i in range(1, time_steps):
        distance.append(np.linalg.norm(trajectory[i, 1:] - 
        trajectory[0, 1:]) + distance[i-1]) 
    return np.array(distance).reshape(time_steps, 1)

def add_mov_moments(trajectory_feature, window = 30 * 5):
    my_df = pd.DataFrame(trajectory_feature)
    avg = np.array(my_df.rolling(window, min_periods= 1).mean())
    std = np.array(my_df.rolling(window, min_periods= 1).std())
    #skewness = np.array(my_df.rolling(window, min_periods= 1).skew())
    
    #Fill nan with mean value: #TODO think about this
    std[np.isnan(std)] = 0
    #skewness[np.isnan(skewness)] = np.nanmean(skewness)

    return avg, std #skewness

def add_derivatives(trajectory: np.array) -> np.array:
    time_steps = trajectory.shape[0]
    dx = [0]
    dy = [0]
    x = trajectory[:, 1]
    y = trajectory[:, 2]

    for i in range(1, time_steps):
        dx.append(x[i] - x[i-1])
        dy.append(y[i] - y[i-1])
    return np.array(dx).reshape([time_steps, 1]), np.array(dy).reshape([time_steps, 1])

def add_features(X: np.array, downsampling_factor, derivatives = True, spe_add = True, acc_add = True, ang_speed = True, trav_dist = False, str_line = False,
    return_names = False):
        names = []
        X = X[::downsampling_factor, :]
        tmp = X
        if return_names:
            names.append('time') #0
            names.append('x') #1
            names.append('y') #2
        if derivatives:
            dx, dy = add_derivatives(X)
            tmp = np.hstack((tmp,  dx, dy))
            mmx, mstdx = add_mov_moments(dx)
            mmy, mstdy = add_mov_moments(dy)
            tmp = np.hstack((tmp, mmx, mstdx, mmy, mstdy))
            if return_names:
                names.append('dx') #3
                names.append('dy') # 4
                names.append('mm_dx') # 5
                names.append('mstd_dx') # 6
                names.append('mm_dy') # 7
                names.append('mstd_dy') # 8
        if spe_add or acc_add:
            speed, acceleration = add_speed_acceleration(X)
        if spe_add:
            tmp = np.hstack((tmp, speed))
            mm, mstd = add_mov_moments(speed)
            tmp = np.hstack((tmp, mm, mstd))
            if return_names:
                names.append('speed') # 9
                names.append('mm_speed') # 10
                names.append('mstd_speed') # 11
        if acc_add:
            tmp = np.hstack((tmp, acceleration))
            mm, mstd = add_mov_moments(acceleration)
            tmp = np.hstack((tmp, mm, mstd))
            if return_names:
                names.append('acceleration') # 12
                names.append('mm_acceleration') # 13
                names.append('mstd_acceleration') # 14
        if ang_speed:
            angular_speed =  add_angular_speed(X)
            tmp = np.hstack((tmp, angular_speed))
            mm, mstd = add_mov_moments(angular_speed)
            tmp = np.hstack((tmp, mm, mstd))
            if return_names:
                names.append('angular_speed') # 15
                names.append('mm_angular_speed') # 16
                names.append('mstd_angular_speed') # 17
        if trav_dist:
            trav_distance = travel_distance(X)
            tmp = np.hstack((tmp, trav_distance))
            mm, mstd = add_mov_moments(trav_distance)
            tmp = np.hstack((tmp, mm, mstd))
            if return_names:
                names.append('trav_distance') # 18
                names.append('mm_trav_distance') # 19
                names.append('mstd_trav_distance') # 20
        if str_line:
            srtg_l = straigth_line(X)
            tmp = np.hstack((tmp, srtg_l))
            mm, mstd = add_mov_moments(srtg_l)
            tmp = np.hstack((tmp, mm, mstd))
            if return_names:
                names.append('straight_line_distance') # 21
                names.append('mm_straight_line_distance') # 22
                names.append('mstd_straight_line_distance') # 23

            assert not (np.isnan(tmp).any())
        
        #print("Values returned, in order: ", names)
        if return_names:
            return tmp.reshape(1, tmp.shape[0], tmp.shape[1]).astype(np.float32), names
        else:
            return tmp.reshape(1, tmp.shape[0], tmp.shape[1]).astype(np.float32)

def normalize_temporal_series(X: np.array):
    
    standardized = zscore(X, axis = 1) # standardizes over the trajectory steps!
    return standardized

def split_train_val_test(X: np.array,
                        G: np.array,
                        Y: np.array,
                        GI: pd.DataFrame,
                        MI: pd.DataFrame,
                        stratify: bool,
                        test = False,
                        val_size = 0.3,
                        test_size = 0.50,
                        random_state = 124)-> tuple: 
    '''
    test: If validation data should be further splitted into test.
    test_size: proportion over train set after validation split.
    ------------------
    Returns tuple with X train, val, (test) and Y train, val, (test)
    '''

    if stratify:
        strat1 = G
    else:
        strat1 = None
    X_train, X_val, G_train, G_val, y_train, y_val, GI_train, GI_val, MI_train, MI_val = \
     train_test_split(  X,
                        G,
                        Y,
                        GI,
                        MI,
                        test_size= val_size,
                        stratify = strat1,
                        random_state=random_state)
    if test:
        if stratify:
            strat2 = G_val
        else:
            strat2 = None
        #print(np.unique(strat2, return_counts= True)[1])
        #print(np.unique(G_train, return_counts= True)[1])
        X_val, X_test, G_val, G_test, y_val, y_test, GI_val, GI_test, MI_val, MI_test = \
             train_test_split(  X_val,
                                G_val,
                                y_val,
                                GI_val,
                                MI_val,
                                test_size= test_size,
                                stratify= strat2,
                                random_state=random_state)

        return (    X_train,
                    X_val,
                    X_test,
                    G_train,
                    G_val,
                    G_test,
                    y_train,
                    y_val,
                    y_test,
                    GI_train,
                    GI_val,
                    GI_test,
                    MI_train,
                    MI_val,
                    MI_test)
    else:

        return( X_train,
                X_val, 
                G_train, 
                G_val, 
                y_train, 
                y_val, 
                GI_train,
                GI_val,
                MI_train,
                MI_val)

def convert_to_image(X: np.array, resolution = (256, 256), ones = False):
    '''
    X -> Data with speed. Assuming speed in position 3.
    This DOWNSAMPLES in a different way. Since the number of available points decreases
    to the resolution*2. If there is a collision the last value is saved, the first one is forgotten.
    
    Returns:
    A gray-scale image. The color is given by the speed of the aphid in each point.
    '''

    # Make the minimum 0. This changes the orientation of the image.
    print('Preparing images...')
    print('Input shape:', X.shape)
   
    images = []
    for i in tqdm(range(X.shape[0])):
        trajectory = X[i]
       # print(trajectory)
        #print(np.max(trajectory[:, 3:4]))
        minimum_x = np.min(trajectory[: , 1])
        minimum_y = np.min(trajectory[: , 2])
        
        trajectory[:, 1] = trajectory[:, 1] - minimum_x
        trajectory[:, 2] = trajectory[:, 2] - minimum_y


        # Make the maximum res -1. This also changes orientation.
        mult_x = (resolution[0] - 1) / np.max(trajectory[: , 1])
        mult_y = (resolution[1] - 1) / np.max(trajectory[: , 2]) 
        trajectory[:, 1] = np.round(trajectory[:, 1] *  mult_x).astype(int)
        trajectory[:, 2] = np.round(trajectory[:, 2] * mult_y).astype(int)
        
        
        # Fill image with the speed intensity relative to the maximum in the arena
        image = np.zeros((1, resolution[0], resolution[1], 1)) #placeholder

        #color = trajectory[:, 3:4]/max_speed
        #color = np.clip(trajectory[:, 3:4]/4, 0, 1) # Ad hoc value based on the frequencies
        color = np.clip(trajectory[:, 3:4]/np.max(trajectory[:, 3:4]), 0, 1).astype(np.float32)

        if ones:
             image[0, trajectory[:, 1].astype(int), trajectory[:, 2].astype(int)] = 1
        else:
            image[0, trajectory[:, 1].astype(int), trajectory[:, 2].astype(int)] = color
        
        images.append(image)

    return np.concatenate(images, axis = 0)