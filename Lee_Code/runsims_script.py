### Run a batch of simulations of reaching task, optionally using different alpha, beta (AKA ratio), g values, then save it with date and time in filename 

from RT_class_LRO import MotorCortex_RT
import numpy as np
import time
from datetime import datetime, date
import pickle

start_time = time.time()
today = (date.today()).strftime("%b-%d-%Y")
tmp_file_name = '/home/lee/reaching_task_model/CCGP_DPCA/Results/results_LRO/results_LRO_temp_' + today + '.pkl'

### CCGP divisions, numbers are quadrants in the target space
NS1 = [0,3,1,2]
NS2 = [1,2,0,3]
EW1 = [0,1,3,2]
EW2 = [3,2,0,1]
ds = [NS1, NS2, EW1, EW2]

### Set parameter values / ranges
N = 1000 # Neurons in net
nTargets = 8 # Targets in task
N_noise = 0 # Add N-dimensional noise, binary
g_min = 1.75
g_max = 1.75
alpha_min = 0.25
alpha_max = 0.25
r_min = 0.5
r_max = 1.5
n_alphas = 1
n_gs = 1
n_nets = 300
n_ratios = 30
n_wins = 1

ratios = np.linspace(r_min, r_max, n_ratios)
gs = np.linspace(g_min, g_max, n_gs)
alphas = np.linspace(alpha_min, alpha_max, n_alphas)
windows = np.array([1])

### Dims for all these arrays are: g, alpha, ratio, rep, win
DPCA_8s = np.zeros([n_gs, n_alphas, n_ratios, n_nets, n_wins]) 
CCGPs = np.zeros([n_gs, n_alphas, n_ratios, n_nets, n_wins]) 
mean_cos_NNs = np.zeros([n_gs, n_alphas, n_ratios, n_nets, n_wins]) 
results_per_div = np.zeros(len(ds))

for nind in range(n_nets):
    for gind, g in enumerate(gs):
        for aind, alpha in enumerate(alphas):
            for rind, ratio in enumerate(ratios):
                print('Running g, a, ratio, rep =', g, alpha, ratio, (nind+1))
                      
                beta = alpha * ratio
                params = {'N':N, 'nTargets': nTargets, 'g':g, 'alpha':alpha, 'beta':beta, 'window':1, 'N_noise':N_noise}
                net = MotorCortex_RT(**params)
                w_lro_fun = net.gen_linear_RO() # Readout weights
                cortex = net.Dynamics(isTest=1, w_ro = w_lro_fun, target_idx=net.task['target_idx'])
                labels_in_sim = net.task['target_idx'][0::net.task['Target_in_len']]                
                for wind, alt_window in enumerate(windows):
                    CMPTI, CMPTI_by_target, CMPT, cortex_quads, DPCA_8 = net.subset_analysis(net.cortex, labels_in_sim, net.subset, window=alt_window)
                    DPCA_8s[gind, aind, rind, nind, wind] = DPCA_8
                      
                    for dind, div in enumerate(ds):
                        result = net.CCGP_test(cortex_quads, div)
                        results_per_div[dind] = result[1]
                    CCGPs[gind, aind, rind, nind, wind] = np.mean(results_per_div)
                    results_per_div = np.zeros(len(ds))                  
      
    ### Save tmp results
    if (nind%9 == 0) & (nind < n_nets-1) & (nind != 0):
        total_time = (time.time() - start_time)
        details = {'gs': gs, 'alphas': alphas,'ratios': ratios, 'n_nets': n_nets, 'n_windows': n_wins, 'N_noise': N_noise, 'result_dims': 'dims of matrices are: g, alpha, ratio, net', 'runtime_sec': total_time, 'CCGP_kernel': 'linear'}
        details['tmp_netnum'] = (nind+1)
        with open(tmp_file_name, 'wb') as tmp_results_file:
            pickle.dump([details, DPCA_8s, CCGPs], tmp_results_file)

total_time = (time.time() - start_time)

### Save results in pkl file named w its creation time
now = datetime.now()
today = date.today()
current_time = now.strftime("%H:%M:%S")
current_date = today.strftime("%b-%d-%Y")
results_file_name = f'results_LRO_' + current_date + '_' + current_time + '.pkl'

with open(results_file_name, 'wb') as results_file:
    pickle.dump([details, DPCA_8s, CCGPs], results_file)