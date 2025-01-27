### Class for reaching task with linear readout 

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
from sklearn.svm import SVC
import pickle
import numpy as np
from numpy.linalg import eig
import random
from numpy.linalg import inv


class MotorCortex_RT:
    def __init__(self, **params):   
        self.__dict__.update(params) # Unpacks all params entries to self.blabla variables
        
        ### Set target coordinates and time constants 
        self.r = 1
        self.dim = 2
        self.Targets = self.Generate_Targets(self.r)
        target_reps = 20 # total repetitions of each target 
        episode_num = target_reps * self.nTargets
        self.tau = 1
        tau = self.tau
        dt = 0.1 * self.tau 
        self.dt = dt
        Target_in_time = 50 * self.tau
        Target_in_len = int(Target_in_time/self.dt) # for convinience, all times in steps
        time = Target_in_time * episode_num
        self.time = time
        steps = int(self.time/self.dt)
        self.noise_sd = 0.1
        self.noise_scale = self.N_noise * 0.1
        noise_ang = np.zeros(steps) 
        highD_noise_change = np.zeros(steps) 
        
        self.tlist, target_idx = self.gen_seqns(target_reps)

        ### Parameters 
        self.phi = np.tanh
        self.thresh = 0  # Neuron activation threshold 
        self.scale = 1 / (np.sqrt(self.N))
        self.train_reps = 10
        self.stop_train = self.train_reps * self.nTargets * Target_in_len # when to stop learning

        ### Connectivity
        self.J_gg = np.random.normal(0, self.g * self.scale, size=(self.N, self.N)) # Inner RNN weights
        self.V = np.random.normal(0, 1, [self.N, self.nTargets]) # High-D Input weights            
        self.u_ang = np.random.normal(0, 1, size=(self.N)) # Theta input weights
        self.u_xy = (1/np.sqrt(2)) * np.random.normal(0, 1, size=(self.N, self.dim)) # XY input weights
        self.w_ro = (1/np.sqrt(2)) * np.random.normal(0, 1, size=(self.N, self.dim)) # Initial readout weights
        self.w_fb = self.u_xy

        ### Saving self variables
        self.task = {'nTargets': self.nTargets, 'Targets': self.Targets, 'Target_in_len': Target_in_len,
                     'target_idx': target_idx, 'tlist': self.tlist,
                     'target_reps': target_reps, 'train_reps':self.train_reps, 'highD_noise_change':highD_noise_change, 'noise_ang':noise_ang}
        self.connectivity = {'J_gg': self.J_gg, 'u': self.u_xy, 'V': self.V, 'w_ro':self.w_ro, 'w_fb':self.w_fb}
        self.sim_params = {'phi': self.phi, 'thresh': self.thresh, 'tau': self.tau, 'dt': self.dt, 'time': time,
                           'g': self.g, 'alpha': self.alpha, 'beta': self.beta}
        self.details = {'N': self.N, 
                   'task': self.task, 'connectivity': self.connectivity, 'sim_params': self.sim_params}
    
    def Generate_Targets(self, r):
        
        ### Set (x,y) targets for task. Input: radius of reach, meaning amplitude of required output
        Targets = []
        ang = (2 * np.pi) / self.nTargets # ang dif between equidistant targets
        for k in range(self.nTargets):
            x1 = r * np.cos(ang * k)
            x2 = r * np.sin(ang * k)
            Targets.append([x1, x2])
        Targets = np.array(Targets)
        Targets[np.abs(Targets) < 0.0005] = 0
        return Targets
        
    def gen_seqns(self, target_reps):     
        
    ### Create target & noise time sequences (binary vectors) and value sequences
    
        max_noise_rad = (2*np.pi / self.nTargets) * 0.2
        episode_num = target_reps * self.nTargets
        Target_in_time = 50 * self.tau
        Target_in_len = int(Target_in_time/self.dt)
        time = Target_in_time * episode_num
        steps = int(time/self.dt)
        tlist_init = np.arange(self.nTargets)
        random.shuffle(tlist_init)
        tlist = tlist_init
        for jj in range(target_reps - 1):
            tlist_init = np.arange(self.nTargets)
            random.shuffle(tlist_init)
            tlist = np.append(tlist, [tlist_init])

        tlist_full = tlist
        target_idx = np.zeros(steps)

        for l in range(episode_num):
            idx1 = l*Target_in_len
            target_idx[idx1:idx1+Target_in_len] = tlist[0]
            tlist = np.delete(tlist,[0])

        target_idx = target_idx.astype(int)
        return tlist_full, target_idx  
    
    def gen_linear_RO(self):
        
        ### Generate weights using linear net approximation
        treps = 20
        tlist, target_idx = self.gen_seqns(treps)
        M = self.Dynamics(0, self.w_ro, target_idx) # Using self.w_ro is fine, there's no FB here since istest=0

        time = 450 # time from the end of each target rep to be saved and used for w_lro, in steps
        total_reps = treps * self.nTargets
        total_time = time * total_reps
        M_lro = np.zeros([self.N, total_time])
        y_lro = np.zeros([total_time, self.dim])
        for rep in range(total_reps):
            idx1 = (rep + 1) * self.task['Target_in_len'] - time
            tind = target_idx[idx1]
            tar = self.Targets[tind]
            M_lro[:, rep * time:(rep + 1) * time] = M[:, idx1:idx1+time]
            y_lro[rep * time:(rep + 1) * time, :] = tar

        P_M = total_time
        MMT1 = inv(np.dot(M_lro,M_lro.T))
        MMT1_M = np.dot(MMT1,M_lro)
        w_lro = np.zeros([self.N, self.dim])
        for d in range(self.dim):
            ys = y_lro[:, d] 
            w_lro[:,d] = MMT1_M @ ys
            
        return w_lro
    
    def Direct_input(self):
        
        ### Examine dynamics with fixed external input, no FB
        Dtime = 500 * self.tau
        Dsteps = int(Dtime/self.dt)
        self.zs_direct = np.zeros([Dsteps, self.dim])
        M = np.zeros([self.N, Dsteps])
        x0 = np.random.normal(0,1,[self.N, 1]) * 0.5 # Initial condition
        x = x0
        tar = np.random.randint(0, self.nTargets)
        ND_I = self.V[:, tar][:, None]
        xy_I = np.sum((self.u_xy * self.Targets[tar]), 1)[:, None]
        ts = np.linspace(0, Dtime-self.dt, Dsteps)
        w_ro_D = (1 / np.sqrt(self.N)) * np.ones([self.N])
        for ii,t in enumerate(ts):
            M[:,ii] = self.phi(x[:,0] - self.thresh)
            dxdt = [1/self.tau] * (-x + np.dot(self.J_gg, self.phi(x - self.thresh)) + \
                                   self.alpha * ND_I + self.beta * xy_I)
            x = x + dxdt * self.dt
            z = np.dot((w_ro_D).T, self.phi(x - self.thresh))
            self.zs_direct[ii, :] = z.squeeze()
        return M, tar


    def Dynamics(self, isTest, w_ro, target_idx):
            
        ### Initial conditions and documentation arrays
        x0 = 0.5 * np.random.normal(0, 1, [self.N, 1]) # Initial condition
        x = x0
        z = np.dot(w_ro.T, self.phi(x - self.thresh))
        e = np.zeros(self.dim)
        steps = target_idx.shape[0]
        time = steps * self.dt
        
        self.z_train = np.zeros([steps, self.dim])
        self.e_train = np.zeros([steps, self.dim])
        self.f = np.zeros([steps, self.dim])
        cortex = np.zeros([self.N, steps])
        cortex_test = np.zeros([self.N, steps - self.stop_train])
        hs = np.zeros([self.N, steps])
        f = np.zeros([steps, 1])
        ts = np.linspace(0, time-self.dt, steps)
        eta = self.noise_scale * np.random.normal(0, self.noise_sd, [self.N, 1])
        
        ### Euler
        for ii,t in enumerate(ts):
            
            cortex[:,ii] = self.phi(x[:,0] - self.thresh)
            hs[:,ii] = x[:,0]
            
            ### High-D input and noise (noise is scaled with external parameter N_noise)
            ND_I = self.V[:,target_idx[ii]] 
                
            ### Low-D input, all options
            original_xy = self.Targets[target_idx[ii]]
            xy_I = np.sum((self.u_xy * original_xy), 1)       
            fb = isTest * np.sum((self.w_fb.T * z), 0)

            dxdt = [1/self.tau] * (-x + np.dot(self.J_gg, self.phi(x)) + eta + self.alpha * ND_I[:, None]
                                   +(1-isTest) * self.beta * xy_I[:, None] + isTest * (self.beta) * fb[:, None])
            x = x + dxdt * self.dt
            self.f[ii] = original_xy
            z = np.dot((w_ro).T, self.phi(x - self.thresh))
            self.z_train[ii, :] = z.squeeze()
            
        self.e_train = self.z_train - self.f        
        self.cortex = cortex
        self.cortex_test = self.cortex[:, self.stop_train:]
        self.hs = hs
        self.subset = 1
        labels_in_sim = target_idx[0::self.task['Target_in_len']]
#         if isTest == 1: # Set to 1 for dimensionality and generalization analysis to be performed
#             ### self variables below are automatic win=1, subset = N calculations, can use different values externally
#             self.CMPTI, self.CMPTI_by_target, self.CMPT, \
#             self.cortex_quads, self.DPCA_8 = self.subset_analysis(self.cortex, labels_in_sim, self.subset, self.window) 
        
        return cortex
    
    def get_CMPTI(self, window, cortex_s, labels_in_sim):
        Target_in_len = self.task['Target_in_len']
        subset_N = cortex_s.shape[0]
        TRs = int(cortex_s.shape[1] / Target_in_len) # total reps
        
        start = 0 # First taget start time
        shift = int(Target_in_len - window - 1) # How long after start of target presentation time to take window
        CMPTI = np.zeros([subset_N, TRs])

        for j in range(TRs):
            idx1 = start + j * Target_in_len  + shift
            CMPTI[:,j] = np.mean(cortex_s[:, idx1:(idx1+window)], 1)        
        CMPTI_by_target = CMPTI[:, labels_in_sim.argsort()[::1]] # Reorder CMPTI by label value
        reshaped_matrix = CMPTI_by_target.reshape(subset_N, self.nTargets, int(TRs/self.nTargets))
        CMPT = np.mean(reshaped_matrix, axis=2)
        return CMPTI, CMPTI_by_target, CMPT
    
    def get_quad_info(self, CMPTI, labels_in_sim):

        ### Reshaping CMPTI to organizie cortex activity by quadrant of relevant target.
        ### E.g. cortex_quads[3,:,:] = all (mean) cortex vecs for targets in quad 4.

        t_in_q = int(self.nTargets / 4) # Targets in each quadrant
        subset_N = CMPTI.shape[0]
        Rs = int(CMPTI.shape[1]/self.nTargets) # reps of each target in input matrix (This will be either target_reps or test_reps)

        ### Rearrange data by quadrant of target.
        cortex_quads = np.zeros([4, subset_N, t_in_q * Rs])
        labels_quads = np.zeros([4, t_in_q * Rs])
        for j in range(4):
            idx = j * t_in_q * Rs
            cortex_quads[j,:,:] = CMPTI[:, idx:idx + t_in_q * Rs]
            labels_quads[j, :] = labels_in_sim[idx:idx + t_in_q * Rs]
        return cortex_quads

    def PCA_cortex(self, cortex, subset):
        
        ### Input: activity array (N-by-steps OR N-by-nTargets), subset ratio (value = int/N)
        cortex_active = cortex
        Crtx = (cortex_active - (cortex_active.mean(axis=1))[:,None]) / cortex_active.std(axis=1)[:,None] # Normalize
        C = Crtx @ Crtx.transpose()
        evals, evecs = np.linalg.eigh(C)
        D_PCA = (np.sum(evals)**2)/(np.sum(evals**2))
        return D_PCA
    
    def CCGP_test(self, cortex_quads, division):
        
        ### Using SVM classifier for CCGP testing. Division input should be of the form [0,3,1,2] etc
        ### Given division array is used as [label1, label2, label1, label2] and [train train test test]
    
        ind1 = division[0]
        ind2 = division[1]
        ind3 = division[2]
        ind4 = division[3]
        CCGP_results = np.zeros([2])

        train_data = np.concatenate((np.squeeze(cortex_quads[ind1,:,:]),np.squeeze(cortex_quads[ind2,:,:])), axis=1).T
        train_labels = np.concatenate((np.zeros(len(train_data)// 2), np.ones(len(train_data) // 2)))
        test_data = np.concatenate((np.squeeze(cortex_quads[ind3,:,:]), np.squeeze(cortex_quads[ind4,:,:])), axis=1).T
        test_labels = np.concatenate((np.zeros(len(test_data)// 2), np.ones(len(test_data) // 2)))

        train_data, train_labels = shuffle(train_data, train_labels)
        test_data, test_labels = shuffle(test_data, test_labels)

        svm = SVC(kernel='linear', C=1.0) # C is 1/margin 
        svm.fit(train_data, train_labels) # Fit linear model with Stochastic Gradient Descent

        predictions_train = svm.predict(train_data)
        predictions_test = svm.predict(test_data)
        train_score = accuracy_score(predictions_train, train_labels)
        test_score = accuracy_score(predictions_test, test_labels)
        svm.score(train_data, train_labels)

        CCGP_results[0] = train_score
        CCGP_results[1] = test_score
        
        return CCGP_results
    
    def subset_analysis(self, cortex_s, labels_in_sim, subset, window):
        ### Get DPCA, CCGP measures, optionally only for some subset = (int/1000) of the network
        ### with temporal averaging for CMPT being done over window timesteps
        
        subset_N = int(subset * self.N)
        if subset_N < self.N:
            idxs = np.random.randint(0, self.N-1, [subset_N])
            cortex_s = cortex_s[idxs, :]
            self.w_ro_s = self.w_ro[idxs] # For examining readout
        
        CMPTI_s, CMPTI_by_target, CMPT_s = self.get_CMPTI(window, cortex_s, labels_in_sim) 
        cortex_quads_s = self.get_quad_info(CMPTI_by_target, labels_in_sim)
        DPCA_s = self.PCA_cortex(cortex_s, subset)
        DPCA_8_s = self.PCA_cortex(CMPT_s, subset)
        
        return CMPTI_s, CMPTI_by_target, CMPT_s, cortex_quads_s, DPCA_8_s