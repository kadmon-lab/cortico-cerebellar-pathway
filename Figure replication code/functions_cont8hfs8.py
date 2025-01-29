import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as clr
from matplotlib.patches import Rectangle

def gaussian(x, sigma, mu=0):
    return np.exp(-((x-mu)**2) / (2*sigma**2))

def make_conv_kernel( sigma= 30):
    nb_time_frames = 6*sigma
    conv_kernel = gaussian(np.arange(-nb_time_frames//2,nb_time_frames//2,1), sigma=sigma)
    conv_kernel /= np.sum(conv_kernel)
    
    return conv_kernel

def make_window_fct(width = 30):
    conv_kernel = 1/width * np.ones(width)
    return conv_kernel

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def get_data(data_file = 'Penny_cont8combined_dic.npy'):
    """
    Returns the dictionnary which keys are 'target', 'mat', 'er' and 'dir' when a force-field is applied 
    
    Each element of the dictionnary is made of lists of length #neurons_recorded
    
    the list corresponding to the 'mat' key contains elements which are arrays of shape (#trials x #time_frames) 
    
    Parameters:
    
    data_file : string indicating the location of the file
    """
    res = np.load(data_file, allow_pickle = True).flat[0]
    return res


def smoothen_data(dic, sigma=30):
    nb_neurons = len(dic['mat'])
    conv_kernel = make_conv_kernel( sigma = sigma)
    smooth_mat = []
    
    for neuron in range(nb_neurons):
        nb_trials = dic['mat'][neuron].shape[0]
        res = []
        
        for trial in range(nb_trials):
            rec = dic['mat'][neuron][trial,:]
            res.append( np.convolve(rec, conv_kernel, mode='same'))        
        smooth_mat.append(np.array(res))
    
    dic['mat'] = smooth_mat
    return dic

def smoothen_data_window_fct(dic, width=30):
    nb_neurons = len(dic['mat'])
    conv_kernel = make_window_fct(width = width)
    smooth_mat = []
    
    for neuron in range(nb_neurons):
        nb_trials = dic['mat'][neuron].shape[0]
        res = []
        
        for trial in range(nb_trials):
            rec = dic['mat'][neuron][trial,:]
            res.append( np.convolve(rec, conv_kernel, mode='same'))        
        smooth_mat.append(np.array(res))
    
    dic['mat'] = smooth_mat
    return dic


def add_condition_average_to_dic(dic):
    """
    Returns:
    
    1: the input dictionnary in which has been added the 'mat_cond_avg' and 'target_cond_avg' keys that 
    contains the neural response average over trials for each target (condition) and the corresponding target.
    
    2: a filter of size #nb_neurons with True if all of 8 targets were recorded for at least one trial for 
    that neuron and False if not.
    """
    nb_time_frames = dic['mat'][0].shape[1]
    nb_neurons = len(dic['mat']) #total number of neurons recorded
    dic['mat_cond_avg'] = []
    dic['target_cond_avg'] = np.arange(1,9)
    neuron_filter = np.zeros(nb_neurons)
    for neuron in range(nb_neurons):
        res = []
        RedFlag = False
        for target in range(1,9):
            res1 = []
            flag = True
            for trial in range(dic['target'][neuron].shape[0]):
                if dic['target'][neuron][trial] == target:
                    flag = False
                    res1.append(dic['mat'][neuron][trial])
            if flag == True : #meaning if not all of the 8 targets were controlled
                RedFlag = True

            res.append(np.array(res1).mean(axis = 0))

        if RedFlag == False:  # if all of the 8 targets were controlled then we save data for the neuron
            dic['mat_cond_avg'].append(np.array(res))
            neuron_filter[neuron] = 1
        else:
            dic['mat_cond_avg'].append(np.zeros((8,nb_time_frames)))
            
            
    return (dic , neuron_filter.astype(dtype = 'bool'))


def add_condition_average_to_dic_with_testset(dic):
    nb_time_frames = dic['mat'][0].shape[1]
    nb_neurons = len(dic['mat'])
    nfilter = np.zeros(nb_neurons)

    nb_tr_per_target = np.zeros((nb_neurons,8))
    for neuron in range(nb_neurons):
        for target in range(8):
            nb_tr_per_target[neuron,target] = (dic['target'][neuron] == (target+1)).sum()

    nfilter = (nb_tr_per_target > 2).all(axis = 1)

    dic['mat_cond_avg'] = []  # Train set
    dic['mat_cond_avg_test'] = [] # Test set

    for neuron in range(nb_neurons):
        res = []
        res_test = []
        if nfilter[neuron] == True:
            for target in range(8):
                res1 = []
                res1_test = []

                tr_inds = np.where(dic['target'][neuron] == (target+1))[0]

                res1_test.append(dic['mat'][neuron][tr_inds[0]])

                for trial in range(1 , int(nb_tr_per_target[neuron,target])):
                        res1.append(dic['mat'][neuron][tr_inds[trial]])

                res.append(np.array(res1).mean(axis = 0))
                res_test.append(np.array(res1_test).mean(axis = 0))

            dic['mat_cond_avg'].append(np.array(res))
            dic['mat_cond_avg_test'].append(np.array(res_test))

        else:
            dic['mat_cond_avg'].append(np.zeros((8,nb_time_frames)))
            dic['mat_cond_avg_test'].append(np.zeros((8,nb_time_frames)))
    return dic, nfilter
        


def add_zscored_to_dic(dic, t, window_size = 10, test_set = False):
    """
    Parameters:
    
    dic : input dictionnary containing a 'mat_cond_avg' key
    
    t : The beginning of the time window on which we want to z-score the neural response accross time and targets (conditions)
    
    window_size : length of the time window, 1 = 30ms
    
    test_set : If True, creates another key specific to the testset
    
    Returns:
    
    input dictionnary on which has been added the corresponding 'mat_zscored' key
    """
    
    if test_set == False:

        dic['mat_zscored'] = []
        for neuron in range(len(dic['mat_cond_avg'])):
            dic['mat_zscored'].append( (dic['mat_cond_avg'][neuron][:,:] - dic['mat_cond_avg'][neuron][:,t:t+window_size].mean())/
                                            (1e-4+dic['mat_cond_avg'][neuron][:,t:t+window_size].std()) )
    elif test_set == True:
        dic['mat_zscored'] = []
        dic['mat_zscored_test'] = []
        
        for neuron in range(len(dic['mat_cond_avg'])):
            dic['mat_zscored'].append( (dic['mat_cond_avg'][neuron][:,:] - dic['mat_cond_avg'][neuron][:,t:t+window_size].mean())/
                                      (1e-4+dic['mat_cond_avg'][neuron][:,t:t+window_size].std()) )
        for neuron in range(len(dic['mat_cond_avg_test'])):
            dic['mat_zscored_test'].append( (dic['mat_cond_avg_test'][neuron][:,:] - dic['mat_cond_avg'][neuron][:,t:t+window_size].mean())/
                                           (1e-4+dic['mat_cond_avg'][neuron][:,t:t+window_size].std()) )
        

    return dic



def TDR_avg_over_trials(z_scored_data, neuron_filter, t , window_size=10, Npca = 12):
    """
    Parameters : 
    
    z_scored_data: input data as given by dic['mat_zscored']
    
    neuron_filter: neuron_filter of neurons for which all of 8 targets were recorded for at least one trial
    
    t: beginning of the time window on which to perform the TDR
    
    window_size: length of the time window on which to perform the TDR
    
    Npca: number of PCs to build the subspace on which to project the regression
    
    Returns:
    
    betas_orth: a matrix made of the TDR axes
    
    """
    nb_neur = neuron_filter.sum() # nb of neurons for which all of 8 targets were recorded for at least one trial
    
    M = np.ones((8,3))
    for i in range(8):
        M[i,1:3] = np.array([np.cos(i*np.pi/4) , np.sin(i*np.pi/4)]) # M is the matrix of size regressors x conditions
        
        
    R = np.array(z_scored_data)[neuron_filter,:,t:t+window_size]
    
    ### Regression
    betas = np.linalg.inv(M.transpose() @ M) @ M.transpose() @ R.transpose() 
    ###
    
    new_beta = np.zeros((3,window_size,nb_neur)) # reordered betas
    for i in range(3):
        for j in range(window_size):
            new_beta[i,j,:] = betas[j,i,:]
            
    ### construction of the matrix projecting on the Npca PCs
    X = R.reshape((nb_neur,window_size*8))
    C = X @ X.transpose()     
    eigvals, eigvects = np.linalg.eigh(C)[0], np.linalg.eigh(C)[1]
    D = np.zeros((nb_neur,nb_neur))
    for i in range(1,Npca+1):
        D += np.outer(eigvects[:,-i], eigvects[:,-i])
    ###
    
    betas_pca = (new_beta@D)
    
    t_max = np.argmax( np.linalg.norm(betas_pca , axis = 2), axis = 1 )
    betas_max = np.zeros((nb_neur,3))
    for i in range(3):
        betas_max[:,i] = betas_pca[i,t_max[i],:]
        
    betas_orth = np.linalg.qr(betas_max)[0]
    
    return betas_orth, t+t_max, eigvals, eigvects


def plot_tdr(data_file = 'Penny_cont8combined_dic.npy', t=85, window_size=10, Npca=12, sigma = 30, center_on_go = True):
    
    """
    Plots the TDR for a given control8 dataset
    """
    nb_time_frames = 6*sigma
    
    dic = get_data(data_file)
    
    dic = smoothen_data(dic, sigma = sigma)
    
    (dic , neuron_filter) = add_condition_average_to_dic(dic)
    
    dic = add_zscored_to_dic(dic, t=t, window_size = window_size)
    
    betas_orth, t_max, eigvals, eigvects = TDR_avg_over_trials(dic['mat_zscored'], neuron_filter, t=t , window_size=window_size, Npca = Npca)
    
    fig, ax = plt.subplots(1, 2, figsize = (10,5), dpi = 200)
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray']    
    zordr = -(np.array([1,3,5,7,6,4,2,0])+50)

    ### reorient the axis so that ordered anti clockwise and first direction is on the right
    X_target1 = np.array(dic['mat_zscored'])[neuron_filter,0,:]
    X_target3 = np.array(dic['mat_zscored'])[neuron_filter,2,:]
    if (betas_orth[:,1]@X_target1[:,t:t+window_size]).mean() < 0:
        betas_orth[:,1] *= -1
    if (betas_orth[:,2]@X_target3[:,t:t+window_size]).mean() < 0:
        betas_orth[:,2] *= -1
    if center_on_go == True:
        if (betas_orth[:,0]@X_target1[:,2400]).mean() < 0:
            betas_orth[:,0] *= -1
    else:
        if (betas_orth[:,0]@X_target1[:,3000]).mean() < 0:
            betas_orth[:,0] *= -1
    
    ###

    for i in range(8):
        X_target = np.array(dic['mat_zscored'])[neuron_filter,i,t:t+window_size]
        ax[0].plot((betas_orth[:,1]@X_target), (betas_orth[:,2]@X_target), '-' , linewidth = 3, alpha = 1,color = colors[i], zorder = zordr[i])
    
    if center_on_go == True:
        
        for i in range(8):
            X_target = np.array(dic['mat_zscored'])[neuron_filter,i,:3001]
            ax[0].plot((betas_orth[:,1]@X_target[:,nb_time_frames//2:]), (betas_orth[:,2]@X_target[:,nb_time_frames//2:]),
                       '-' , linewidth = 1, alpha = 0.8,label=i+1, zorder = zordr[i]) ## [:,nb_time_frames//2:] to avoid the border effects on the smoothing

            ax[0].scatter((betas_orth[:,1]@X_target[:,2400]), (betas_orth[:,2]@X_target[:,2400]) ,color = colors[i], 
                        marker = 'o', s = 20 , zorder=100, edgecolors = 'k')
            ax[0].scatter((betas_orth[:,1]@X_target[:,3000]), (betas_orth[:,2]@X_target[:,3000]) ,color = colors[i],
                        marker = 'o', s = 20 , zorder=100)

        ax[0].scatter((betas_orth[:,1]@X_target[:,2400]), (betas_orth[:,2]@X_target[:,2400]) ,color = colors[i], 
                    marker = 'o', s = 20 , zorder=100, edgecolors = 'k', label = 'Cue signal')
        ax[0].scatter((betas_orth[:,1]@X_target[:,3000]), (betas_orth[:,2]@X_target[:,3000]) ,color = colors[i], 
                    marker = 'o', s = 20 , zorder=100, label = 'Go signal')    
    else:
        for i in range(8):
            X_target = np.array(dic['mat_zscored'])[neuron_filter,i,:3601]
            ax[0].plot((betas_orth[:,1]@X_target[:,nb_time_frames//2:]), (betas_orth[:,2]@X_target[:,nb_time_frames//2:]),
                       '-' , linewidth = 1, alpha = 0.8,label=i+1, zorder = zordr[i]) ## [:,nb_time_frames//2:] to avoid the border effects on the smoothing

            ax[0].scatter((betas_orth[:,1]@X_target[:,3000]), (betas_orth[:,2]@X_target[:,3000]) ,color = colors[i], marker = 'o', s = 20 , zorder=100, edgecolors = 'k')
            ax[0].scatter((betas_orth[:,1]@X_target[:,3600]), (betas_orth[:,2]@X_target[:,3600]) ,color = colors[i], marker = 'o', s = 20 , zorder=100)
        
        ax[0].scatter((betas_orth[:,1]@X_target[:,3000]), (betas_orth[:,2]@X_target[:,3000]) ,color = colors[i], marker = 'o', s = 20 , zorder=100, edgecolors = 'k', label = 'Cue signal')
        ax[0].scatter((betas_orth[:,1]@X_target[:,3600]), (betas_orth[:,2]@X_target[:,3600]) ,color = colors[i], marker = 'o', s = 20 , zorder=100, label = 'Go signal')
        
    origin = np.array([-14.8, -15])
    xhat = np.array([3, 0])
    yhat = np.array([0, 3])

    # Plotting 2 unit vectors
    ax[0].arrow(*origin, *xhat, head_width=0.2, color='k')
    ax[0].arrow(*origin, *yhat, head_width=0.2, color='k')
    ax[0].text(-14, -16.2, r'$\beta_x^\perp$', size = 10)
    ax[0].text(-16.4, -14, r'$\beta_y^\perp$', size = 10)
    ax[0].legend(fontsize=6)
    
    
    for i in range(8):
        X_target = np.array(dic['mat_zscored'])[neuron_filter,i,t:t+window_size]
        ax[1].plot((betas_orth[:,1]@X_target), (betas_orth[:,0]@X_target), '-' , linewidth = 3, alpha = 1,color = colors[i], zorder = zordr[i])
    
    if center_on_go == True:

        for i in range(8):
            X_target = np.array(dic['mat_zscored'])[neuron_filter,i,:3001]
            ax[1].plot((betas_orth[:,1]@X_target[:,nb_time_frames//2:]), (betas_orth[:,0]@X_target[:,nb_time_frames//2:]), 
                       '-' , linewidth = 1, alpha = 0.8,label=i+1, color = colors[i], zorder = zordr[i])


            ax[1].scatter((betas_orth[:,1]@X_target[:,2400]), (betas_orth[:,0]@X_target[:,2400]) ,color = colors[i], 
                        marker = 'o', s = 20 , zorder=100, edgecolors = 'k')
            ax[1].scatter((betas_orth[:,1]@X_target[:,3000]), (betas_orth[:,0]@X_target[:,3000]) ,color = colors[i],
                        marker = 'o', s = 20 , zorder=100)

        ax[1].scatter((betas_orth[:,1]@X_target[:,2400]), (betas_orth[:,0]@X_target[:,2400]) ,color = colors[i], 
                    marker = 'o', s = 20 , zorder=100, edgecolors = 'k', label = 'Cue signal')
        ax[1].scatter((betas_orth[:,1]@X_target[:,3000]), (betas_orth[:,0]@X_target[:,3000]) ,color = colors[i], 
                    marker = 'o', s = 20 , zorder=100, label = 'Go signal')    
    else:
        for i in range(8):
            X_target = np.array(dic['mat_zscored'])[neuron_filter,i,:3601]
            ax[1].plot((betas_orth[:,1]@X_target[:,nb_time_frames//2:]), (betas_orth[:,0]@X_target[:,nb_time_frames//2:]), 
                       '-' , linewidth = 1, alpha = 0.8,label=i+1, color = colors[i], zorder = zordr[i])


            ax[1].scatter((betas_orth[:,1]@X_target[:,3000]), (betas_orth[:,0]@X_target[:,3000]) ,color = colors[i], marker = 'o', s = 20 , zorder=100, edgecolors = 'k')
            ax[1].scatter((betas_orth[:,1]@X_target[:,3600]), (betas_orth[:,0]@X_target[:,3600]) ,color = colors[i],marker = 'o', s = 20 , zorder=100)

        ax[1].scatter((betas_orth[:,1]@X_target[:,3000]), (betas_orth[:,0]@X_target[:,3000]) ,color = colors[i], marker = 'o', s = 20 , zorder=100, edgecolors = 'k', label = 'Cue signal')
        ax[1].scatter((betas_orth[:,1]@X_target[:,3600]), (betas_orth[:,0]@X_target[:,3600]) ,color = colors[i], marker = 'o', s = 20 , zorder=100, label = 'Go signal')    
        

    origin = np.array([-14.8, -15])
    xhat = np.array([3, 0])
    yhat = np.array([0, 3])

    # Plotting 2 unit vectors
    ax[1].arrow(*origin, *xhat, head_width=0.2, color='k')
    ax[1].arrow(*origin, *yhat, head_width=0.2, color='k')
    ax[1].text(-14, -16.1, r'$\beta_x^\perp$', size = 10)
    ax[1].text(-16.4, -14, r'$\beta_0^\perp$', size = 10)
    ax[1].legend(fontsize=6)
    
    return()   


def add_zscored_trials_to_dic(dic, t, window_size = 10):
    
    """
    Returns a dictionnary in which has been added the key 'mat_zscored_trials' which has the same shape
    as dic['mat'] except every trial has been zscored accross a time window and all trials.
    
    Parameter:
    
    dic: input dictionnary containing the ['mat'] ['target'] and ['mat_cond_avg'] keys
    
    t: beginning of the time window on which the zscored
    
    window_size: size of the time window
    """
    nb_time_frames = dic['mat'][0].shape[1]
    nb_neurons = len(dic['mat'])

    dic['mat_zscored_trials'] = []

    for neuron in range(nb_neurons):
        data_mat = np.array(dic['mat'], dtype = 'object')[neuron]
        nb_trials = data_mat.shape[0]

        res = np.zeros((nb_trials,nb_time_frames))

        for trial in range(nb_trials):
            if data_mat[0].shape[0] > 0:
                target = np.array(dic['target'],dtype='object')[neuron][trial]
                res[trial,:] =((data_mat[trial] - dic['mat_cond_avg'][neuron][:,t:t+window_size].mean())/
                               (1e-10+dic['mat_cond_avg'][neuron][:,t:t+window_size].std()) )

        dic['mat_zscored_trials'].append(res)

    dic['mat_zscored_trials'] = np.array(dic['mat_zscored_trials'], dtype = 'object')
    
    return dic



def add_std_over_trials_to_dic(dic, t_max):
    
    """
    Returns a dictionnary in which has been added the 'mat_trials_std' key 
    of shape (nb_neurons x 8 x 2) that contains the std over trials for each neuron
    for each target at two times which are the 2nd and third entry of t_max
    
    Parameters:
    
    dic: entry dictionnary
    
    t_max: t_max as defined in the TDR method
    
    """
    
    nb_neurons = len(dic['mat'])
    dic['mat_trials_std'] = np.zeros((nb_neurons, 8, 2))

    for neuron in range(nb_neurons):
        for target in range(9):
            res = []
            nb_trials = dic['mat_zscored_trials'][neuron].shape[0]
            if nb_trials >1:
                for trial in range(nb_trials):
                    if dic['target'][neuron][trial] == target:
                        res.append(dic['mat_zscored_trials'][neuron][trial,t_max[1:3]])

            dic['mat_trials_std'][neuron,target-1,:] = np.array(res).std(axis = 0)      

    return dic


def get_filter_for_learned_target(dic):
    
    nb_neurons = len(dic['target'])
    neuron_filter_learned_target = np.zeros((nb_neurons,8))
    for n in range(nb_neurons):
        if dic['target'][n].shape[0] > 1:
            target = dic['target'][n][0] - 1
            neuron_filter_learned_target[n,target] = True
            
    return neuron_filter_learned_target


def get_filter_for_direction(dic):

    nb_neurons = len(dic['target'])
    neuron_filter_dir = np.zeros((nb_neurons,8))
    for n in range(nb_neurons):
        if dic['target'][n].shape[0] > 1:
            target = dic['target'][n][0] - 1
            neuron_filter_dir[n,target] = dic['dir'][n][0,0]

    return neuron_filter_dir


def add_zscored_wrt_cont8_to_dic(dic, dic_cont8, t, window_size = 10):
    
    """
    Returns a dictionnary in which has been added the key 'mat_zscored_wrt_cont8' which has the same shape
    as dic['mat'] except every trial has been zscored accross a time window and all trials.
    
    Parameter:
    
    dic: input dictionnary containing the ['mat'] keys
    
    dic_cont8: dictionnary containing the ['target'] and ['mat_cond_avg'] keys 
    
    t: beginning of the time window on which the zscored
    
    window_size: size of the time window
    """
    nb_time_frames = dic_cont8['mat'][0].shape[1]
    nb_neurons = len(dic['mat'])

    dic['mat_zscored_wrt_cont8'] = []

    for neuron in range(nb_neurons):
        data_mat = np.array(dic['mat'], dtype = 'object')[neuron]
        nb_trials = data_mat.shape[0]

        res = np.zeros((nb_trials,nb_time_frames))

        for trial in range(nb_trials):
            if data_mat[0].shape[0] > 0:
                target = np.array(dic['target'],dtype='object')[neuron][trial]
                res[trial,:] =((data_mat[trial] - dic_cont8['mat_cond_avg'][neuron][:,t:t+window_size].mean())/
                               (1e-10+dic_cont8['mat_cond_avg'][neuron][:,t:t+window_size].std()) )

        dic['mat_zscored_wrt_cont8'].append(res)

    dic['mat_zscored_wrt_cont8'] = np.array(dic['mat_zscored_wrt_cont8'], dtype = 'object')
    
    return dic


def add_zscored_wrt_cont8_truncated_to_dic(dic, neuron_filter, nb_time_frames):
    nb_n = neuron_filter.sum()
    nb_neurons = dic['mat_zscored_wrt_cont8'].shape[0]

    res1 = np.zeros(nb_n)
    for n in range(nb_n):
        res1[n] = np.array(dic['mat_zscored_wrt_cont8'])[neuron_filter][n].shape[0]
    nb_trials = int(res1.min())

    dic['mat_zscored_wrt_cont8_truncated'] = []
    dic['mat_zscored_wrt_cont8_truncated'] = np.zeros((nb_neurons, nb_trials, nb_time_frames))

    inds = np.where(neuron_filter == True)[0]
    for n in range(nb_n):
        data_mat = np.array(dic['mat_zscored_wrt_cont8'], dtype = 'object')[inds[n]]
        dic['mat_zscored_wrt_cont8_truncated'][inds[n]] = data_mat[:nb_trials,:]
        
    return dic, nb_trials
        
    
    
    
def perform_PCAs(dic_tot, dic_cont8, dic_hfs, neuron_filter, t, window_size):
    """
    Perform PCA on the given data dictionaries using the specified neuron filter, time window, and time offset.

    Args:
        dic_tot (dict): Data dictionary containing total data.
        dic_cont8 (dict): Data dictionary containing control data.
        dic_hfs (dict): Data dictionary containing HFS data.
        neuron_filter (ndarray): Boolean filter for selecting neurons.
        t (int): Time offset for selecting data.
        window_size (int): Window size for selecting data.

    Returns:
        A tuple containing three arrays of eigenvalues and eigenvectors for total, control, and HFS data, respectively.
    """

    # Extract z-scored data for total data
    z_scored_data = dic_tot['mat_zscored']

    # Compute number of selected neurons
    nb_neur = neuron_filter.sum()

    # Select and reshape data for total data
    R = np.array(z_scored_data)[neuron_filter, :, t:t+window_size]
    X = R.reshape((nb_neur, window_size*16))

    # Compute covariance matrix and its eigenvalues/eigenvectors for total data
    C = X @ X.transpose()
    eigvals, eigvects = np.linalg.eigh(C)[0], np.linalg.eigh(C)[1]

    # Extract z-scored data for control data
    z_scored_data_cont8 = dic_cont8['mat_zscored']

    # Select and reshape data for control data
    R = np.array(z_scored_data_cont8)[neuron_filter, :, t:t+window_size]
    X = R.reshape((nb_neur, window_size*8))

    # Compute covariance matrix and its eigenvalues/eigenvectors for control data
    C = X @ X.transpose()
    eigvals_cont, eigvects_cont = np.linalg.eigh(C)[0], np.linalg.eigh(C)[1]

    # Extract z-scored data for HFS data
    z_scored_data_hfs = dic_hfs['mat_zscored']

    # Select and reshape data for HFS data
    R = np.array(z_scored_data_hfs)[neuron_filter, :, t:t+window_size]
    X = R.reshape((nb_neur, window_size*8))

    # Compute covariance matrix and its eigenvalues/eigenvectors for HFS data
    C = X @ X.transpose()
    eigvals_hfs, eigvects_hfs = np.linalg.eigh(C)[0], np.linalg.eigh(C)[1]

    return eigvals, eigvects, eigvals_cont, eigvects_cont, eigvals_hfs, eigvects_hfs

def perform_PCA(dic_cont8, neuron_filter, t, window_size):
    """
    Perform PCA on the given data dictionaries using the specified neuron filter, time window, and time offset.

    Args:
        dic_cont8 (dict): Data dictionary containing control data.
        neuron_filter (ndarray): Boolean filter for selecting neurons.
        t (int): Time offset for selecting data.
        window_size (int): Window size for selecting data.

    Returns:
        A tuple containing three arrays of eigenvalues and eigenvectors for total, control, and HFS data, respectively.
    """

    # Compute number of selected neurons
    nb_neur = neuron_filter.sum()

    # Extract z-scored data for control data
    z_scored_data_cont8 = dic_cont8['mat_zscored']

    # Select and reshape data for control data
    R = np.array(z_scored_data_cont8)[neuron_filter, :, t:t+window_size]
    X = R.reshape((nb_neur, window_size*8))

    # Compute covariance matrix and its eigenvalues/eigenvectors for control data
    C = X @ X.transpose()
    eigvals_cont, eigvects_cont = np.linalg.eigh(C)[0], np.linalg.eigh(C)[1]

    return eigvals_cont, eigvects_cont



def get_rand_tr_indices_same_nbt_per_target(dic_cont8, dic_hfs):

    nb_neurons = len(dic_hfs['target'])

    trial_indices_to_consider_hfs = np.zeros(nb_neurons, dtype = 'object')
    trial_indices_to_consider_cont8 = np.zeros(nb_neurons, dtype = 'object')

    for neuron in range(nb_neurons):

        hfs_targets = dic_hfs['target'][neuron]
        cont8_targets = dic_cont8['target'][neuron]

        hfs_trial_indices = np.array([])
        cont8_trial_indices = np.array([]) 
        for target in range(1,9):
            # get the lower value between number of trials for this given target in HFS and Cont8 data
            min_nbt_for_target = np.min( [(hfs_targets == target).sum() , (cont8_targets == target).sum()]) 

            # choose min_nbt_for_target indices corresponding to the selected target for each condition 
            hfs_indices = np.random.choice(np.where(hfs_targets == target)[0], size = min_nbt_for_target, replace = False)
            cont8_indices = np.random.choice(np.where(cont8_targets == target)[0], size = min_nbt_for_target, replace = False)

            # have all these indices be in one list
            hfs_trial_indices=np.concatenate((hfs_trial_indices, hfs_indices))
            cont8_trial_indices=np.concatenate((cont8_trial_indices, cont8_indices))


        trial_indices_to_consider_hfs[neuron] = hfs_trial_indices.astype('int')
        trial_indices_to_consider_cont8[neuron] = cont8_trial_indices.astype('int')
        
    return (trial_indices_to_consider_cont8, trial_indices_to_consider_hfs)




def add_condition_average_to_dic_with_defined_trials(dic, trial_indices_to_consider):
    """
    Returns:
    
    1: the input dictionnary in which has been added the 'mat_cond_avg' and 'target_cond_avg' keys that 
    contains the neural response average over trials for each target (condition) and the corresponding target.
    
    2: a filter of size #nb_neurons with True if all of 8 targets were recorded for at least one trial for 
    that neuron and False if not.
    """
    nb_time_frames = dic['mat'][0].shape[1]
    nb_neurons = len(dic['mat']) #total number of neurons recorded
    dic['mat_cond_avg'] = []
    dic['target_cond_avg'] = np.arange(1,9)
    neuron_filter = np.zeros(nb_neurons)
    for neuron in range(nb_neurons):
        res = []
        RedFlag = False
        for target in range(1,9):
            res1 = []
            flag = True
            for trial in trial_indices_to_consider[neuron]:
                if dic['target'][neuron][trial] == target:
                    flag = False
                    res1.append(dic['mat'][neuron][trial])
            if flag == True : # meaning if not all of the 8 targets were controlled
                RedFlag = True

            res.append(np.array(res1).mean(axis = 0))

        if RedFlag == False:  # if all of the 8 targets were controlled then we save data for the neuron
            dic['mat_cond_avg'].append(np.array(res))
            neuron_filter[neuron] = 1
        else:
            dic['mat_cond_avg'].append(np.zeros((8,nb_time_frames)))
            
            
    return (dic , neuron_filter.astype(dtype = 'bool'))