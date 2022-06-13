#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_selection import SelectKBest
import warnings
import pandas as pd
from pywt import wavedec
from pywt import waverec
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import VotingClassifier
import pywt
import scipy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import biosppy as bp
#from keras.utils import plot_model #plot_model(model, to_file='model.png')
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SVMSMOTE
import pylab as pl
from sklearn.utils import resample
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from biosppy.signals import tools


# In[2]:


eeg1 = pd.read_csv("train_eeg1.csv").iloc[:, 1:]
eeg2 = pd.read_csv("train_eeg2.csv").iloc[:, 1:]
emg  = pd.read_csv("train_emg.csv").iloc[:, 1:]
df_y = pd.read_csv("train_labels.csv").iloc[:, 1:]




# ## Brief description of all the features that will be extracted
# Mean power of all frequencies within the band and power maximum (i.e. power of the dominant frequency) for each of the ten frequency bands.
#
# We further calculated the ratio of theta band (4.5–8.5 Hz) power divided by delta band (1–4 Hz) power which is frequently used in sleep scoring, including commercial programs based on threshold formulas.
#
# From band pass filtered PAA data the following four parameters were calculated for each of the ten frequency bands:
#
# 1) number of baseline crossings per second;
#
# 2) durationbetween two consecutive baseline crossings;
#
# 3) integrated amplitude of half-waves; and
#
# 4) maximal amplitude of half-waves.
#
# We also calculated the ratio between integrated halfwave-amplitudes in the theta band and in the delta range, respectively.
#
# Further, we calculated a parameter for regularity of rhythmic activity for each of the ten frequency bands.
#
# This rhythm score was computed as the number of baseline crossings per second multiplied by the interval between consecutive zero crossings. Oscillations with stable frequency during the entire epoch length would lead to a rhythm score of 1.
#
# The integrated amplitude of the neck muscle EMG com-pleted the set of parameters, resulting in a total of 73 variables

# In[3]:


eeg1 = eeg1.values
eeg2 = eeg2.values
emg = emg.values
y = df_y.values




# In[4]:


def concat_epochs(eeg1, eeg2, emg):
    num_epochs = len(eeg1)
    
    eeg1_transformed = []
    eeg2_transformed = []
    emg_transformed = []
    
    for i in range(0, num_epochs):
        if i == num_epochs - 1:
            epoch1, epoch2  = i-1, i
        else:
            epoch1, epoch2 = i, i+1
        
        eeg1_epochs = np.concatenate((eeg1[epoch1], eeg1[epoch2]))
        eeg1_transformed.append(eeg1_epochs)
    
        eeg2_epochs = np.concatenate((eeg2[epoch1], eeg2[epoch2]))
        eeg2_transformed.append(eeg2_epochs)
    
        emg_epochs = np.concatenate((emg[epoch1], emg[epoch2]))
        emg_transformed.append(emg_epochs)
    
    return np.array(eeg1_transformed), np.array(eeg2_transformed), np.array(emg_transformed)

eeg1_8, eeg2_8, emg_8  = concat_epochs(eeg1, eeg2, emg)

#eeg1_8, eeg2_8, emg_8 = segment(eeg1, eeg2, emg)   # _8 short for 8-seconds
print(eeg1_8.shape)
print(eeg2_8.shape)
print(emg_8.shape)


# ### Part 1 of 2 (feature extraction): Power spectral analysis

# In[5]:


# The neck muscle EMG activity was high pass filtered (30–70 Hz)


# In[6]:


def filter_emg(row):
    filtered = tools.filter_signal(row, ftype = 'butter', band = 'highpass', order = 8, frequency = 30, sampling_rate = 128)
    return filtered[0]
filtered_emg = np.apply_along_axis(filter_emg, axis = 1, arr = emg_8)



# In[7]:


#print(filtered_emg.shape)


# In[8]:



def get_freq_in_range(band, row):
    if (band == 'delta'):  #  case 1
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [1, 4], sampling_rate = 128)
        return res[0]
    elif (band  == 'theta'): # case 2
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [4.5, 8.5], sampling_rate = 128)
        return res[0]
    elif (band == 'theta1'): # case 3
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [4.5, 6.5], sampling_rate = 128)
        return res[0]
    elif (band == 'theta2'): # case 4
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [7, 8.5], sampling_rate = 128)
        return res[0]
    elif (band == 'sigma'): # case 5
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [9, 14], sampling_rate = 128)
        return res[0]
    elif (band == 'beta1'): # case 6
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [14.5, 18.5], sampling_rate = 128)
        return res[0]
    elif (band == 'beta2'): # case 7
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [19, 30], sampling_rate = 128)
        return res[0]
    elif (band == 'gamma1'): # case 8
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [30.5, 48], sampling_rate = 128)
        return res[0]
    elif (band == 'gamma2'): # case 9
        res = tools.filter_signal(row, ftype = 'butter', band = 'highpass', order = 8, frequency = 52, sampling_rate = 128)
        return res[0]
    elif (band == 'tot'): # case 10
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [1, 63.5], sampling_rate = 128)
        return res[0]
    


# In[9]:




# In[10]:


delta_freq =  np.apply_along_axis(lambda x: get_freq_in_range('delta', x), axis = 1, arr = eeg1_8)
theta_freq =  np.apply_along_axis(lambda x: get_freq_in_range('theta', x), axis = 1, arr = eeg1_8)
theta1_freq = np.apply_along_axis(lambda x: get_freq_in_range('theta1', x), axis = 1, arr = eeg1_8)
theta2_freq = np.apply_along_axis(lambda x: get_freq_in_range('theta2', x), axis = 1, arr = eeg1_8)
sigma_freq =  np.apply_along_axis(lambda x: get_freq_in_range('sigma', x), axis = 1, arr = eeg1_8)
beta1_freq  =  np.apply_along_axis(lambda x: get_freq_in_range('beta1', x), axis = 1, arr = eeg1_8)
beta2_freq = np.apply_along_axis(lambda x: get_freq_in_range('beta2', x), axis = 1, arr = eeg1_8)
gamma1_freq = np.apply_along_axis(lambda x: get_freq_in_range('gamma1', x), axis = 1, arr = eeg1_8)
gamma2_freq = np.apply_along_axis(lambda x: get_freq_in_range('gamma2', x), axis = 1, arr = eeg1_8)
tot_freq    = np.apply_along_axis(lambda x: get_freq_in_range('tot', x), axis = 1, arr = eeg1_8)


delta_freq2 =  np.apply_along_axis(lambda x: get_freq_in_range('delta', x), axis = 1, arr = eeg2_8)
theta_freq2 =  np.apply_along_axis(lambda x: get_freq_in_range('theta', x), axis = 1, arr = eeg2_8)
theta1_freq2 = np.apply_along_axis(lambda x: get_freq_in_range('theta1', x), axis = 1, arr = eeg2_8)
theta2_freq2 = np.apply_along_axis(lambda x: get_freq_in_range('theta2', x), axis = 1, arr = eeg2_8)
sigma_freq2 =  np.apply_along_axis(lambda x: get_freq_in_range('sigma', x), axis = 1, arr = eeg2_8)
beta1_freq2  =  np.apply_along_axis(lambda x: get_freq_in_range('beta1', x), axis = 1, arr = eeg2_8)
beta2_freq2 = np.apply_along_axis(lambda x: get_freq_in_range('beta2', x), axis = 1, arr = eeg2_8)
gamma1_freq2 = np.apply_along_axis(lambda x: get_freq_in_range('gamma1', x), axis = 1, arr = eeg2_8)
gamma2_freq2 = np.apply_along_axis(lambda x: get_freq_in_range('gamma2', x), axis = 1, arr = eeg2_8)
tot_freq2    = np.apply_along_axis(lambda x: get_freq_in_range('tot', x), axis = 1, arr = eeg2_8)


# In[11]:


def get_power_max(row):
    res = tools.welch_spectrum(row, sampling_rate= 128)
    return [np.mean(res[1]), np.max(res[1])]
power_feat_delta = np.apply_along_axis(get_power_max, axis = 1, arr = delta_freq)
power_feat_theta = np.apply_along_axis(get_power_max, axis = 1, arr = theta_freq)
power_feat_theta1 = np.apply_along_axis(get_power_max, axis = 1, arr = theta1_freq)
power_feat_theta2 = np.apply_along_axis(get_power_max, axis = 1, arr = theta2_freq)
power_feat_sigma = np.apply_along_axis(get_power_max, axis = 1, arr = sigma_freq)
power_feat_beta1 = np.apply_along_axis(get_power_max, axis = 1, arr = beta1_freq)
power_feat_beta2 = np.apply_along_axis(get_power_max, axis = 1, arr = beta2_freq)
power_feat_gamma1 = np.apply_along_axis(get_power_max, axis = 1, arr = gamma1_freq)
power_feat_gamma2 = np.apply_along_axis(get_power_max, axis = 1, arr = gamma2_freq)
power_feat_tot = np.apply_along_axis(get_power_max, axis = 1, arr = tot_freq)

power_feat_delta2 = np.apply_along_axis(get_power_max, axis = 1, arr = delta_freq2)
power_feat_theta2 = np.apply_along_axis(get_power_max, axis = 1, arr = theta_freq2)
power_feat_theta12 = np.apply_along_axis(get_power_max, axis = 1, arr = theta1_freq2)
power_feat_theta22 = np.apply_along_axis(get_power_max, axis = 1, arr = theta2_freq2)
power_feat_sigma2 = np.apply_along_axis(get_power_max, axis = 1, arr = sigma_freq2)
power_feat_beta12 = np.apply_along_axis(get_power_max, axis = 1, arr = beta1_freq2)
power_feat_beta22 = np.apply_along_axis(get_power_max, axis = 1, arr = beta2_freq2)
power_feat_gamma12 = np.apply_along_axis(get_power_max, axis = 1, arr = gamma1_freq2)
power_feat_gamma22 = np.apply_along_axis(get_power_max, axis = 1, arr = gamma2_freq2)
power_feat_tot2 = np.apply_along_axis(get_power_max, axis = 1, arr = tot_freq2)


# In[12]:





pow_mn_delta = power_feat_delta[:, 0]
pow_mn_theta = power_feat_theta[:, 0]

pow_mn_delta2 = power_feat_delta2[:, 0]
pow_mn_theta2 = power_feat_theta2[:, 0]
#pow_mn_theta1 = power_feat_theta1[:, 0]
#pow_mn_theta2 = power_feat_theta2[:, 0]
#pow_mn_sigma = power_feat_sigma[:, 0]
#pow_mn_beta1 = power_feat_beta1[:, 0]
#pow_mn_beta2 = power_feat_beta2[:, 0]
#pow_mn_gamma1 = power_feat_gamma1[:, 0]
#pow_mn_gamma2 = power_feat_gamma2[:, 0]
#pow_mn_tot = power_feat_tot[:, 0]

#pow_mx_delta = power_feat_delta[:, 1]
#pow_mx_theta = power_feat_theta[:, 1]
#pow_mx_theta1 = power_feat_theta1[:, 1]
#pow_mx_theta2 = power_feat_theta2[:, 1]
#pow_mx_sigma = power_feat_sigma[:, 1]
#pow_mx_beta1 = power_feat_beta1[:, 1]
#pow_mx_beta2 = power_feat_beta2[:, 1]
#pow_mx_gamma1 = power_feat_gamma1[:, 1]
#pow_mx_gamma2 = power_feat_gamma2[:, 1]
#pow_mx_tot = power_feat_tot[:, 1]


#pow_mn_theta1_t = power_feat_theta1_t[:, 0]
#pow_mn_theta2_t = power_feat_theta2_t[:, 0]
#pow_mn_sigma_t = power_feat_sigma_t[:, 0]
#pow_mn_beta1_t = power_feat_beta1_t[:, 0]
#pow_mn_beta2_t = power_feat_beta2_t[:, 0]
#pow_mn_gamma1_t = power_feat_gamma1_t[:, 0]
#pow_mn_gamma2_t = power_feat_gamma2_t[:, 0]
#pow_mn_tot_t = power_feat_tot_t[:, 0]

#pow_mx_delta_t = power_feat_delta_t[:, 1]
#pow_mx_theta_t = power_feat_theta_t[:, 1]
#pow_mx_theta1_t = power_feat_theta1_t[:, 1]
#pow_mx_theta2_t = power_feat_theta2_t[:, 1]
#pow_mx_sigma_t = power_feat_sigma_t[:, 1]
#pow_mx_beta1_t = power_feat_beta1_t[:, 1]
#pow_mx_beta2_t = power_feat_beta2_t[:, 1]
#pow_mx_gamma1_t = power_feat_gamma1_t[:, 1]
#pow_mx_gamma2_t = power_feat_gamma2_t[:, 1]
#pow_mx_tot_t = power_feat_tot_t[:, 1]


# In[14]:


# theta divided by delta
#print(pow_mx_delta_t.flatten().shape)
#print(eeg1_t.shape)
theta_delta =  pow_mn_theta / pow_mn_delta


theta_delta2 =  pow_mn_theta2 / pow_mn_delta2


#print(theta_delta[0])
#print(pow_mn_theta[0])
#print(pow_mn_delta[0])


# ###  PAA

# In[15]:


# delta -> 1 - 4.5
# theta - > 4 - 10
# theta1    4 - 10
# theta2    4 - 10
# sigma     7 - 20
# beta1     10 - 40
# beta2     10 - 40
# gamma1    25 - 80
# gamma2    25 - 80
# tot       1 - 80

def preprocess_sig(band, row):
    if (band == 'delta'):  #  case 1
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [1, 4.5], sampling_rate = 128)
        return res[0]
    elif (band  == 'theta'): # case 2
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [4, 10], sampling_rate = 128)
        return res[0]
    elif (band == 'sigma'): # case 5
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [7, 20], sampling_rate = 128)
        return res[0]
    elif (band == 'beta'): # case 6
        res = tools.filter_signal(row, ftype = 'butter', band = 'bandpass', order = 8, frequency = [10, 40], sampling_rate = 128)
        return res[0]
    elif (band == 'gamma'): # case 8
        res = tools.filter_signal(row, ftype = 'butter', band = 'highpass', order = 8, frequency = 25, sampling_rate = 128)
        return res[0]
    elif (band == 'tot'): # case 10
        res = tools.filter_signal(row, ftype = 'butter', band = 'highpass', order = 8, frequency = 1, sampling_rate = 128)
        return res[0]


# In[16]:




delta_pre = np.apply_along_axis(lambda x: preprocess_sig('delta', x), axis = 1, arr= eeg1_8)
theta_pre = np.apply_along_axis(lambda x: preprocess_sig('theta', x), axis = 1, arr= eeg1_8)
sigma_pre = np.apply_along_axis(lambda x: preprocess_sig('sigma', x), axis = 1, arr= eeg1_8)
beta_pre = np.apply_along_axis(lambda x: preprocess_sig('beta', x), axis = 1, arr= eeg1_8)
gamma_pre = np.apply_along_axis(lambda x: preprocess_sig('gamma', x), axis = 1, arr= eeg1_8)
tot_pre = np.apply_along_axis(lambda x: preprocess_sig('tot', x), axis = 1, arr= eeg1_8)


delta_pre2 = np.apply_along_axis(lambda x: preprocess_sig('delta', x), axis = 1, arr= eeg2_8)
theta_pre2 = np.apply_along_axis(lambda x: preprocess_sig('theta', x), axis = 1, arr= eeg2_8)
sigma_pre2 = np.apply_along_axis(lambda x: preprocess_sig('sigma', x), axis = 1, arr= eeg2_8)
beta_pre2 = np.apply_along_axis(lambda x: preprocess_sig('beta', x), axis = 1, arr= eeg2_8)
gamma_pre2 = np.apply_along_axis(lambda x: preprocess_sig('gamma', x), axis = 1, arr= eeg2_8)
tot_pre2 = np.apply_along_axis(lambda x: preprocess_sig('tot', x), axis = 1, arr= eeg2_8)


# In[17]:



# In[19]:


delta_paa =  np.apply_along_axis(lambda x: get_freq_in_range('delta', x), axis = 1, arr = delta_pre)
theta_paa =  np.apply_along_axis(lambda x: get_freq_in_range('theta', x), axis = 1, arr = theta_pre)
theta1_paa = np.apply_along_axis(lambda x: get_freq_in_range('theta1', x), axis = 1, arr = theta_pre)
theta2_paa = np.apply_along_axis(lambda x: get_freq_in_range('theta2', x), axis = 1, arr = theta_pre)
sigma_paa =  np.apply_along_axis(lambda x: get_freq_in_range('sigma', x), axis = 1, arr = sigma_pre)
beta1_paa  =  np.apply_along_axis(lambda x: get_freq_in_range('beta1', x), axis = 1, arr = beta_pre)
beta2_paa = np.apply_along_axis(lambda x: get_freq_in_range('beta2', x), axis = 1, arr = beta_pre)
gamma1_paa = np.apply_along_axis(lambda x: get_freq_in_range('gamma1', x), axis = 1, arr = gamma_pre)
gamma2_paa = np.apply_along_axis(lambda x: get_freq_in_range('gamma2', x), axis = 1, arr = gamma_pre)
tot_paa    = np.apply_along_axis(lambda x: get_freq_in_range('tot', x), axis = 1, arr = tot_pre)


delta_paa2 =  np.apply_along_axis(lambda x: get_freq_in_range('delta', x), axis = 1, arr = delta_pre2)
theta_paa2 =  np.apply_along_axis(lambda x: get_freq_in_range('theta', x), axis = 1, arr = theta_pre2)
theta1_paa2 = np.apply_along_axis(lambda x: get_freq_in_range('theta1', x), axis = 1, arr = theta_pre2)
theta2_paa2 = np.apply_along_axis(lambda x: get_freq_in_range('theta2', x), axis = 1, arr = theta_pre2)
sigma_paa2 =  np.apply_along_axis(lambda x: get_freq_in_range('sigma', x), axis = 1, arr = sigma_pre2)
beta1_paa2  =  np.apply_along_axis(lambda x: get_freq_in_range('beta1', x), axis = 1, arr = beta_pre2)
beta2_paa2 = np.apply_along_axis(lambda x: get_freq_in_range('beta2', x), axis = 1, arr = beta_pre2)
gamma1_paa2 = np.apply_along_axis(lambda x: get_freq_in_range('gamma1', x), axis = 1, arr = gamma_pre2)
gamma2_paa2 = np.apply_along_axis(lambda x: get_freq_in_range('gamma2', x), axis = 1, arr = gamma_pre2)
tot_paa2    = np.apply_along_axis(lambda x: get_freq_in_range('tot', x), axis = 1, arr = tot_pre2)

#delta_paa = delta_paa    # to prevent overflows later
#delta_paa_t = delta_paa_t


# 1) number of baseline crossings per second;
#
# 2) duration between two consecutive baseline crossings;
#
# 3) integrated amplitude of half-waves; and
#
# 4) maximal amplitude of half-waves.

# In[20]:


iter = 0
def paa_features(row):
    #global iter
    #print(row)
    #zc2  = tools.zero_cross(signal = row, detrend=False)
    
    df = np.diff(np.sign(row))
    zc = np.nonzero(np.abs(df) > 0)[0]
    
    if (len(zc) <= 1):  # i.e. there is only 1 zero crossing
        #print('empty')
        #iter += 1
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    
    
    rate = len(zc)/len(row)
    differences = np.ediff1d(zc)
    avg_dur = np.mean(differences)
    std_dur = np.std(differences)  # regularity of the rhythm
    # integrated amplitude of half-wave
    # we get the amplitudes beginning from right after the index of the zero and upto and including the index of the next zero
    med_max_ampl = []
    mean_int_ampl = []
    for idx, el in enumerate(zc):
        if (idx >= (len(zc)-1)):
            break
        el2 = zc[idx+1]
        rng = np.arange(el+1, el2+1)
        half_wave = row[rng]
        area_approx = np.sum(half_wave**2)
        #print(area_approx)
        max_ampl = np.max(half_wave)
        med_max_ampl.append(max_ampl)
        mean_int_ampl.append(area_approx)
        
    #print(len(med_max_ampl))
    #print(row)
        
    var3 = np.max(med_max_ampl)
    var4 = np.median(med_max_ampl)
    var5 = np.mean(mean_int_ampl)
    
    res = np.array([rate, avg_dur, std_dur, var3 , var4 , var5 ], dtype = 'float64')
    
    return res
 


# In[ ]:


def paa_features_delta(row):
    row_dec = row/1e100
    max_amp = np.max(row_dec)
    min_amp = np.min(row_dec)
    med_amp = np.mean(row_dec)
    diffs = np.ediff1d(row_dec)
    min_slope = np.min(diffs)
    max_slope = np.max(diffs)
    largest_slope = 1
    if (int(np.sign(min_slope)) == int(np.sign(max_slope))):
        largest_slope = int(np.sign(min_slope))*max(np.abs(max_slope), np.abs(min_slope))
    elif (np.abs(min_slope) > np.abs(max_slope)):
        largest_slope = int(np.sign(min_slope))*np.abs(min_slope)
    else:
        largest_slope = np.abs(max_slope)
        
    area_approx = np.sum(row_dec**2)
    avg_slope = np.mean(diffs)
    res = np.array([max_amp, min_amp, largest_slope, area_approx, avg_slope, med_amp])
    return res
    


# In[21]:


delta_paa_feat = np.apply_along_axis(paa_features_delta, axis = 1, arr = delta_paa)
theta_paa_feat = np.apply_along_axis(paa_features, axis = 1, arr = theta_paa)
theta1_paa_feat = np.apply_along_axis(paa_features, axis = 1, arr = theta1_paa)
theta2_paa_feat = np.apply_along_axis(paa_features, axis = 1, arr = theta2_paa)
sigma_paa_feat = np.apply_along_axis(paa_features, axis = 1, arr = sigma_paa)
beta1_paa_feat = np.apply_along_axis(paa_features, axis = 1, arr = beta1_paa)
beta2_paa_feat = np.apply_along_axis(paa_features, axis = 1, arr = beta2_paa)
gamma1_paa_feat = np.apply_along_axis(paa_features, axis = 1, arr = gamma1_paa)
gamma2_paa_feat = np.apply_along_axis(paa_features, axis = 1, arr = gamma2_paa)
tot_paa_feat = np.apply_along_axis(paa_features, axis = 1, arr = tot_paa)


delta_paa_feat2 = np.apply_along_axis(paa_features_delta, axis = 1, arr = delta_paa2)
theta_paa_feat2 = np.apply_along_axis(paa_features, axis = 1, arr = theta_paa2)
theta1_paa_feat2 = np.apply_along_axis(paa_features, axis = 1, arr = theta1_paa2)
theta2_paa_feat2 = np.apply_along_axis(paa_features, axis = 1, arr = theta2_paa2)
sigma_paa_feat2 = np.apply_along_axis(paa_features, axis = 1, arr = sigma_paa2)
beta1_paa_feat2 = np.apply_along_axis(paa_features, axis = 1, arr = beta1_paa2)
beta2_paa_feat2 = np.apply_along_axis(paa_features, axis = 1, arr = beta2_paa2)
gamma1_paa_feat2 = np.apply_along_axis(paa_features, axis = 1, arr = gamma1_paa2)
gamma2_paa_feat2 = np.apply_along_axis(paa_features, axis = 1, arr = gamma2_paa2)
tot_paa_feat2 = np.apply_along_axis(paa_features, axis = 1, arr = tot_paa2)


# In[22]:


#print(delta_paa_feat[20:40, :])
# if it prints too many missing values (np.nan) maybe try subtracting the mean of the signal from the signal and then finding the zeros
imp_d = SimpleImputer(missing_values=np.nan, strategy='mean') # d short for delta
imp_t = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_t1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_t2 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_s = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_b1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_b2 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_g1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_g2 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_tot = SimpleImputer(missing_values=np.nan, strategy='mean')


imp_d2 = SimpleImputer(missing_values=np.nan, strategy='mean') # d short for delta
imp_t2 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_t12 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_t22 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_s2 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_b12 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_b22 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_g12 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_g22 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_tot2 = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_d.fit(delta_paa_feat)
imp_t.fit(theta_paa_feat)
imp_t1.fit(theta1_paa_feat)
imp_t2.fit(theta2_paa_feat)
imp_s.fit(sigma_paa_feat)
imp_b1.fit(beta1_paa_feat)
imp_b2.fit(beta2_paa_feat)
imp_g1.fit(gamma1_paa_feat)
imp_g2.fit(gamma2_paa_feat)
imp_tot.fit(tot_paa_feat)


imp_d2.fit(delta_paa_feat2)
imp_t2.fit(theta_paa_feat2)
imp_t12.fit(theta1_paa_feat2)
imp_t22.fit(theta2_paa_feat2)
imp_s2.fit(sigma_paa_feat2)
imp_b12.fit(beta1_paa_feat2)
imp_b22.fit(beta2_paa_feat2)
imp_g12.fit(gamma1_paa_feat2)
imp_g22.fit(gamma2_paa_feat2)
imp_tot2.fit(tot_paa_feat2)


# In[23]:


# In[24]:


#import matplotlib.pyplot as plt
#plt.plot(theta_paa[0])
#plt.ylabel('some numbers')
#plt.show()


# In[25]:


#import matplotlib.pyplot as plt
#plt.plot(delta_paa[575])
#plt.ylabel('some numbers')
#plt.show()


# In[26]:


delta_paa_imputed = imp_d.transform(delta_paa_feat)
theta_paa_imputed =imp_t.transform(theta_paa_feat)
theta1_paa_imputed =imp_t1.transform(theta1_paa_feat)
theta2_paa_imputed =imp_t2.transform(theta2_paa_feat)
sigma_paa_imputed =imp_s.transform(sigma_paa_feat)
beta1_paa_imputed =imp_b1.transform(beta1_paa_feat)
beta2_paa_imputed =imp_b2.transform(beta2_paa_feat)
gamma1_paa_imputed =imp_g1.transform(gamma1_paa_feat)
gamma2_paa_imputed =imp_g2.transform(gamma2_paa_feat)
tot_paa_imputed =imp_tot.transform(tot_paa_feat)


delta_paa_imputed2 = imp_d2.transform(delta_paa_feat2)
theta_paa_imputed2 =imp_t2.transform(theta_paa_feat2)
theta1_paa_imputed2 =imp_t12.transform(theta1_paa_feat2)
theta2_paa_imputed2 =imp_t22.transform(theta2_paa_feat2)
sigma_paa_imputed2 =imp_s2.transform(sigma_paa_feat2)
beta1_paa_imputed2 =imp_b12.transform(beta1_paa_feat2)
beta2_paa_imputed2 =imp_b22.transform(beta2_paa_feat2)
gamma1_paa_imputed2 =imp_g12.transform(gamma1_paa_feat2)
gamma2_paa_imputed2 =imp_g22.transform(gamma2_paa_feat2)
tot_paa_imputed2 =imp_tot2.transform(tot_paa_feat2)


# In[27]:


#print(tot_paa_imputed_t.shape)


# In[28]:


def get_integrated_amp(row):
    return np.sum(row**2)
emg_integrated_amp = np.apply_along_axis(get_integrated_amp, axis =1, arr = filtered_emg)



# In[29]:


# finally ratio of integrated amplitude of theta range to int. amplitude of delta range
ratio_amplitude = theta_paa_imputed[:, 5]/delta_paa_imputed[:, 5]


ratio_amplitude2 = theta_paa_imputed2[:, 5]/delta_paa_imputed2[:, 5]


# In[30]:


# to summarize the feature matrices are
# ratio_amplitude
# delta_paa_imputed, theta_paa_imputed, ... , tot_paa_imputed
# pow_mn_delta, pow_mn_theta, ... , pow_mn_tot
# pow_mx_delta, pow_mx_theta, ... , pow_mx_tot
# theta_delta
# emg_integrated_amp

# The test data's features matrices have exactly the same name but with _t  appended to the matrix identifiers

len1 = len(ratio_amplitude)
len2 = len(theta_delta)
len3 = len(emg_integrated_amp)

ratio_amplitude = ratio_amplitude.reshape((len1, 1))


theta_delta = theta_delta.reshape((len2, 1))


ratio_amplitude2 = ratio_amplitude2.reshape((-1, 1))
theta_delta2 = theta_delta2.reshape((-1, 1))
emg_integrated_amp = emg_integrated_amp.reshape((len3, 1))


# In[ ]:


# standard scale everything
X_train = np.hstack((ratio_amplitude, delta_paa_imputed, theta_paa_imputed, theta1_paa_imputed,
                    theta2_paa_imputed, sigma_paa_imputed, beta1_paa_imputed, beta2_paa_imputed,
                    gamma1_paa_imputed, gamma2_paa_imputed, delta_paa_imputed, tot_paa_imputed,
                     power_feat_delta, power_feat_theta, power_feat_theta1, power_feat_theta2,
                     power_feat_sigma, power_feat_beta1, power_feat_beta2, power_feat_gamma1,
                     power_feat_gamma2, power_feat_tot, theta_delta, emg_integrated_amp,
                     ratio_amplitude2, delta_paa_imputed2, theta_paa_imputed2, theta1_paa_imputed2,
                     theta2_paa_imputed2, sigma_paa_imputed2, beta1_paa_imputed2, beta2_paa_imputed2,
                     gamma1_paa_imputed2, gamma2_paa_imputed2, delta_paa_imputed2, tot_paa_imputed2,
                      power_feat_delta2, power_feat_theta2, power_feat_theta12, power_feat_theta22,
                      power_feat_sigma2, power_feat_beta12, power_feat_beta22, power_feat_gamma12,
                      power_feat_gamma22, power_feat_tot2, theta_delta2
                    ))
                    
del ratio_amplitude, delta_paa_imputed, theta_paa_imputed, theta1_paa_imputed, theta2_paa_imputed, sigma_paa_imputed, beta1_paa_imputed, beta2_paa_imputed

del gamma1_paa_imputed, gamma2_paa_imputed, delta_paa_imputed, tot_paa_imputed, power_feat_delta, power_feat_theta, power_feat_theta1, power_feat_theta2

del power_feat_sigma, power_feat_beta1, power_feat_beta2, power_feat_gamma1, power_feat_gamma2, power_feat_tot, theta_delta, emg_integrated_amp, ratio_amplitude2

del delta_paa_imputed2, theta_paa_imputed2, theta1_paa_imputed2, theta2_paa_imputed2, sigma_paa_imputed2, beta1_paa_imputed2, beta2_paa_imputed2
 
 
del gamma1_paa_imputed2, gamma2_paa_imputed2, delta_paa_imputed2, tot_paa_imputed2, power_feat_delta2, power_feat_theta2, power_feat_theta12, power_feat_theta22
  
del power_feat_sigma2, power_feat_beta12, power_feat_beta22, power_feat_gamma12, power_feat_gamma22, power_feat_tot2, theta_delta2




sc = StandardScaler()
sc = sc.fit(X_train)
X_train = sc.transform(X_train)
y_train = y

def saveResults(best_param, best_score, clft):
    f = open(clft +"_best_params_and_score.txt", "a")
    f.write(str(best_param))
    f.write("\n score: ")
    f.write(str(best_score))
    f.close()

# ****************************************************************

eeg1_t = pd.read_csv("test_eeg1.csv").iloc[:, 1:]
eeg2_t = pd.read_csv("test_eeg2.csv").iloc[:, 1:]
emg_t = pd.read_csv("test_emg.csv").iloc[:, 1:]

eeg1_t = eeg1_t.values
eeg2_t = eeg2_t.values
emg_t = emg_t.values


eeg1_8test, eeg2_8test, emg_8test = concat_epochs(eeg1_t, eeg2_t, emg_t)


filtered_emg_t = np.apply_along_axis(filter_emg, axis = 1, arr = emg_8test)

delta_freq_t =  np.apply_along_axis(lambda x: get_freq_in_range('delta', x), axis = 1, arr = eeg1_8test)
theta_freq_t =  np.apply_along_axis(lambda x: get_freq_in_range('theta', x), axis = 1, arr = eeg1_8test)
theta1_freq_t = np.apply_along_axis(lambda x: get_freq_in_range('theta1', x), axis = 1, arr = eeg1_8test)
theta2_freq_t = np.apply_along_axis(lambda x: get_freq_in_range('theta2', x), axis = 1, arr = eeg1_8test)
sigma_freq_t =  np.apply_along_axis(lambda x: get_freq_in_range('sigma', x), axis = 1, arr = eeg1_8test)
beta1_freq_t  =  np.apply_along_axis(lambda x: get_freq_in_range('beta1', x), axis = 1, arr = eeg1_8test)
beta2_freq_t = np.apply_along_axis(lambda x: get_freq_in_range('beta2', x), axis = 1, arr = eeg1_8test)
gamma1_freq_t = np.apply_along_axis(lambda x: get_freq_in_range('gamma1', x), axis = 1, arr = eeg1_8test)
gamma2_freq_t = np.apply_along_axis(lambda x: get_freq_in_range('gamma2', x), axis = 1, arr = eeg1_8test)
tot_freq_t    = np.apply_along_axis(lambda x: get_freq_in_range('tot', x), axis = 1, arr = eeg1_8test)



delta_freq2_t =  np.apply_along_axis(lambda x: get_freq_in_range('delta', x), axis = 1, arr = eeg2_8test)
theta_freq2_t =  np.apply_along_axis(lambda x: get_freq_in_range('theta', x), axis = 1, arr = eeg2_8test)
theta1_freq2_t = np.apply_along_axis(lambda x: get_freq_in_range('theta1', x), axis = 1, arr = eeg2_8test)
theta2_freq2_t = np.apply_along_axis(lambda x: get_freq_in_range('theta2', x), axis = 1, arr = eeg2_8test)
sigma_freq2_t =  np.apply_along_axis(lambda x: get_freq_in_range('sigma', x), axis = 1, arr = eeg2_8test)
beta1_freq2_t  =  np.apply_along_axis(lambda x: get_freq_in_range('beta1', x), axis = 1, arr = eeg2_8test)
beta2_freq2_t = np.apply_along_axis(lambda x: get_freq_in_range('beta2', x), axis = 1, arr = eeg2_8test)
gamma1_freq2_t = np.apply_along_axis(lambda x: get_freq_in_range('gamma1', x), axis = 1, arr = eeg2_8test)
gamma2_freq2_t = np.apply_along_axis(lambda x: get_freq_in_range('gamma2', x), axis = 1, arr = eeg2_8test)
tot_freq2_t    = np.apply_along_axis(lambda x: get_freq_in_range('tot', x), axis = 1, arr = eeg2_8test)



## try using welch instead of power_spectrum
power_feat_delta_t = np.apply_along_axis(get_power_max, axis = 1, arr = delta_freq_t)
power_feat_theta_t = np.apply_along_axis(get_power_max, axis = 1, arr = theta_freq_t)
power_feat_theta1_t = np.apply_along_axis(get_power_max, axis = 1, arr = theta1_freq_t)
power_feat_theta2_t = np.apply_along_axis(get_power_max, axis = 1, arr = theta2_freq_t)
power_feat_sigma_t = np.apply_along_axis(get_power_max, axis = 1, arr = sigma_freq_t)
power_feat_beta1_t = np.apply_along_axis(get_power_max, axis = 1, arr = beta1_freq_t)
power_feat_beta2_t = np.apply_along_axis(get_power_max, axis = 1, arr = beta2_freq_t)
power_feat_gamma1_t = np.apply_along_axis(get_power_max, axis = 1, arr = gamma1_freq_t)
power_feat_gamma2_t = np.apply_along_axis(get_power_max, axis = 1, arr = gamma2_freq_t)
power_feat_tot_t = np.apply_along_axis(get_power_max, axis = 1, arr = tot_freq_t)

power_feat_delta2_t = np.apply_along_axis(get_power_max, axis = 1, arr = delta_freq2_t)
power_feat_theta2_t = np.apply_along_axis(get_power_max, axis = 1, arr = theta_freq2_t)
power_feat_theta12_t = np.apply_along_axis(get_power_max, axis = 1, arr = theta1_freq2_t)
power_feat_theta22_t = np.apply_along_axis(get_power_max, axis = 1, arr = theta2_freq2_t)
power_feat_sigma2_t = np.apply_along_axis(get_power_max, axis = 1, arr = sigma_freq2_t)
power_feat_beta12_t = np.apply_along_axis(get_power_max, axis = 1, arr = beta1_freq2_t)
power_feat_beta22_t = np.apply_along_axis(get_power_max, axis = 1, arr = beta2_freq2_t)
power_feat_gamma12_t = np.apply_along_axis(get_power_max, axis = 1, arr = gamma1_freq2_t)
power_feat_gamma22_t = np.apply_along_axis(get_power_max, axis = 1, arr = gamma2_freq2_t)
power_feat_tot2_t = np.apply_along_axis(get_power_max, axis = 1, arr = tot_freq2_t)


# In[13]:

pow_mn_delta_t = power_feat_delta_t[:, 0]
pow_mn_theta_t = power_feat_theta_t[:, 0]

pow_mn_delta2_t = power_feat_delta2_t[:, 0]
pow_mn_theta2_t = power_feat_theta2_t[:, 0]

theta_delta_t = pow_mn_theta_t / pow_mn_delta_t
theta_delta2_t = pow_mn_theta2_t / pow_mn_delta2_t





delta_pre_t = np.apply_along_axis(lambda x: preprocess_sig('delta', x), axis = 1, arr= eeg1_8test)
theta_pre_t = np.apply_along_axis(lambda x: preprocess_sig('theta', x), axis = 1, arr= eeg1_8test)
sigma_pre_t = np.apply_along_axis(lambda x: preprocess_sig('sigma', x), axis = 1, arr= eeg1_8test)
beta_pre_t = np.apply_along_axis(lambda x: preprocess_sig('beta', x), axis = 1, arr= eeg1_8test)
gamma_pre_t = np.apply_along_axis(lambda x: preprocess_sig('gamma', x), axis = 1, arr= eeg1_8test)
tot_pre_t = np.apply_along_axis(lambda x: preprocess_sig('tot', x), axis = 1, arr= eeg1_8test)


delta_pre2_t = np.apply_along_axis(lambda x: preprocess_sig('delta', x), axis = 1, arr= eeg2_8test)
theta_pre2_t = np.apply_along_axis(lambda x: preprocess_sig('theta', x), axis = 1, arr= eeg2_8test)
sigma_pre2_t = np.apply_along_axis(lambda x: preprocess_sig('sigma', x), axis = 1, arr= eeg2_8test)
beta_pre2_t = np.apply_along_axis(lambda x: preprocess_sig('beta', x), axis = 1, arr= eeg2_8test)
gamma_pre2_t = np.apply_along_axis(lambda x: preprocess_sig('gamma', x), axis = 1, arr= eeg2_8test)
tot_pre2_t = np.apply_along_axis(lambda x: preprocess_sig('tot', x), axis = 1, arr= eeg2_8test)


# In[18]:


delta_paa_t =  np.apply_along_axis(lambda x: get_freq_in_range('delta', x), axis = 1, arr = delta_pre_t)
theta_paa_t =  np.apply_along_axis(lambda x: get_freq_in_range('theta', x), axis = 1, arr = theta_pre_t)
theta1_paa_t = np.apply_along_axis(lambda x: get_freq_in_range('theta1', x), axis = 1, arr = theta_pre_t)
theta2_paa_t = np.apply_along_axis(lambda x: get_freq_in_range('theta2', x), axis = 1, arr = theta_pre_t)
sigma_paa_t =  np.apply_along_axis(lambda x: get_freq_in_range('sigma', x), axis = 1, arr = sigma_pre_t)
beta1_paa_t  =  np.apply_along_axis(lambda x: get_freq_in_range('beta1', x), axis = 1, arr = beta_pre_t)
beta2_paa_t = np.apply_along_axis(lambda x: get_freq_in_range('beta2', x), axis = 1, arr = beta_pre_t)
gamma1_paa_t = np.apply_along_axis(lambda x: get_freq_in_range('gamma1', x), axis = 1, arr = gamma_pre_t)
gamma2_paa_t = np.apply_along_axis(lambda x: get_freq_in_range('gamma2', x), axis = 1, arr = gamma_pre_t)
tot_paa_t    = np.apply_along_axis(lambda x: get_freq_in_range('tot', x), axis = 1, arr = tot_pre_t)


delta_paa2_t =  np.apply_along_axis(lambda x: get_freq_in_range('delta', x), axis = 1, arr = delta_pre2_t)
theta_paa2_t =  np.apply_along_axis(lambda x: get_freq_in_range('theta', x), axis = 1, arr = theta_pre2_t)
theta1_paa2_t = np.apply_along_axis(lambda x: get_freq_in_range('theta1', x), axis = 1, arr = theta_pre2_t)
theta2_paa2_t = np.apply_along_axis(lambda x: get_freq_in_range('theta2', x), axis = 1, arr = theta_pre2_t)
sigma_paa2_t =  np.apply_along_axis(lambda x: get_freq_in_range('sigma', x), axis = 1, arr = sigma_pre2_t)
beta1_paa2_t  =  np.apply_along_axis(lambda x: get_freq_in_range('beta1', x), axis = 1, arr = beta_pre2_t)
beta2_paa2_t = np.apply_along_axis(lambda x: get_freq_in_range('beta2', x), axis = 1, arr = beta_pre2_t)
gamma1_paa2_t = np.apply_along_axis(lambda x: get_freq_in_range('gamma1', x), axis = 1, arr = gamma_pre2_t)
gamma2_paa2_t = np.apply_along_axis(lambda x: get_freq_in_range('gamma2', x), axis = 1, arr = gamma_pre2_t)
tot_paa2_t    = np.apply_along_axis(lambda x: get_freq_in_range('tot', x), axis = 1, arr = tot_pre2_t)








delta_paa_feat_t = np.apply_along_axis(paa_features_delta, axis = 1, arr = delta_paa_t)
theta_paa_feat_t = np.apply_along_axis(paa_features, axis = 1, arr = theta_paa_t)
theta1_paa_feat_t = np.apply_along_axis(paa_features, axis = 1, arr = theta1_paa_t)
theta2_paa_feat_t = np.apply_along_axis(paa_features, axis = 1, arr = theta2_paa_t)
sigma_paa_feat_t = np.apply_along_axis(paa_features, axis = 1, arr = sigma_paa_t)
beta1_paa_feat_t = np.apply_along_axis(paa_features, axis = 1, arr = beta1_paa_t)
beta2_paa_feat_t = np.apply_along_axis(paa_features, axis = 1, arr = beta2_paa_t)
gamma1_paa_feat_t = np.apply_along_axis(paa_features, axis = 1, arr = gamma1_paa_t)
gamma2_paa_feat_t = np.apply_along_axis(paa_features, axis = 1, arr = gamma2_paa_t)
tot_paa_feat_t = np.apply_along_axis(paa_features, axis = 1, arr = tot_paa_t)


delta_paa_feat2_t = np.apply_along_axis(paa_features_delta, axis = 1, arr = delta_paa2_t)
theta_paa_feat2_t = np.apply_along_axis(paa_features, axis = 1, arr = theta_paa2_t)
theta1_paa_feat2_t = np.apply_along_axis(paa_features, axis = 1, arr = theta1_paa2_t)
theta2_paa_feat2_t = np.apply_along_axis(paa_features, axis = 1, arr = theta2_paa2_t)
sigma_paa_feat2_t = np.apply_along_axis(paa_features, axis = 1, arr = sigma_paa2_t)
beta1_paa_feat2_t = np.apply_along_axis(paa_features, axis = 1, arr = beta1_paa2_t)
beta2_paa_feat2_t = np.apply_along_axis(paa_features, axis = 1, arr = beta2_paa2_t)
gamma1_paa_feat2_t = np.apply_along_axis(paa_features, axis = 1, arr = gamma1_paa2_t)
gamma2_paa_feat2_t = np.apply_along_axis(paa_features, axis = 1, arr = gamma2_paa2_t)
tot_paa_feat2_t = np.apply_along_axis(paa_features, axis = 1, arr = tot_paa2_t)




delta_paa_imputed_t = imp_d.transform(delta_paa_feat_t)
theta_paa_imputed_t =imp_t.transform(theta_paa_feat_t)
theta1_paa_imputed_t =imp_t1.transform(theta1_paa_feat_t)
theta2_paa_imputed_t =imp_t2.transform(theta2_paa_feat_t)
sigma_paa_imputed_t =imp_s.transform(sigma_paa_feat_t)
beta1_paa_imputed_t =imp_b1.transform(beta1_paa_feat_t)
beta2_paa_imputed_t =imp_b2.transform(beta2_paa_feat_t)
gamma1_paa_imputed_t =imp_g1.transform(gamma1_paa_feat_t)
gamma2_paa_imputed_t =imp_g2.transform(gamma2_paa_feat_t)
tot_paa_imputed_t =imp_tot.transform(tot_paa_feat_t)

delta_paa_imputed2_t = imp_d.transform(delta_paa_feat2_t)
theta_paa_imputed2_t =imp_t.transform(theta_paa_feat2_t)
theta1_paa_imputed2_t =imp_t1.transform(theta1_paa_feat2_t)
theta2_paa_imputed2_t =imp_t2.transform(theta2_paa_feat2_t)
sigma_paa_imputed2_t =imp_s.transform(sigma_paa_feat2_t)
beta1_paa_imputed2_t =imp_b1.transform(beta1_paa_feat2_t)
beta2_paa_imputed2_t =imp_b2.transform(beta2_paa_feat2_t)
gamma1_paa_imputed2_t =imp_g1.transform(gamma1_paa_feat2_t)
gamma2_paa_imputed2_t =imp_g2.transform(gamma2_paa_feat2_t)
tot_paa_imputed2_t =imp_tot.transform(tot_paa_feat2_t)



emg_integrated_amp_t = np.apply_along_axis(get_integrated_amp, axis =1, arr = filtered_emg_t)

ratio_amplitude_t = theta_paa_imputed_t[:, 5]/delta_paa_imputed_t[:, 5]
ratio_amplitude2_t = theta_paa_imputed2_t[:, 5]/delta_paa_imputed2_t[:, 5]
theta_delta_t = theta_delta_t.reshape((-1, 1))


ratio_amplitude_t = ratio_amplitude_t.reshape((-1, 1))


ratio_amplitude2_t = ratio_amplitude2_t.reshape((-1, 1))
theta_delta2_t = theta_delta2_t.reshape((-1, 1))
emg_integrated_amp_t = emg_integrated_amp_t.reshape((-1, 1))

                    
X_test = np.hstack((ratio_amplitude_t, delta_paa_imputed_t, theta_paa_imputed_t, theta1_paa_imputed_t,
theta2_paa_imputed_t, sigma_paa_imputed_t, beta1_paa_imputed_t, beta2_paa_imputed_t,
gamma1_paa_imputed_t, gamma2_paa_imputed_t, delta_paa_imputed_t, tot_paa_imputed_t,
 power_feat_delta_t, power_feat_theta_t, power_feat_theta1_t,
 power_feat_sigma_t, power_feat_beta1_t, power_feat_beta2_t, power_feat_gamma1_t,
 power_feat_gamma2_t, power_feat_tot_t, theta_delta_t, emg_integrated_amp_t,
 ratio_amplitude2_t, delta_paa_imputed2_t, theta_paa_imputed2_t, theta1_paa_imputed2_t,
 theta2_paa_imputed2_t, sigma_paa_imputed2_t, beta1_paa_imputed2_t, beta2_paa_imputed2_t,
 gamma1_paa_imputed2_t, gamma2_paa_imputed2_t, delta_paa_imputed2_t, tot_paa_imputed2_t,
  power_feat_delta2_t, power_feat_theta2_t, power_feat_theta12_t, power_feat_theta22_t,
  power_feat_sigma2_t, power_feat_beta12_t, power_feat_beta22_t, power_feat_gamma12_t,
  power_feat_gamma22_t, power_feat_tot2_t, theta_delta2_t
))



del ratio_amplitude_t, delta_paa_imputed_t, theta_paa_imputed_t, theta1_paa_imputed_t
del theta2_paa_imputed_t, sigma_paa_imputed_t, beta1_paa_imputed_t, beta2_paa_imputed_t
del gamma1_paa_imputed_t, gamma2_paa_imputed_t, tot_paa_imputed_t
del power_feat_delta_t, power_feat_theta_t, power_feat_theta1_t
del power_feat_sigma_t, power_feat_beta1_t, power_feat_beta2_t, power_feat_gamma1_t
del power_feat_gamma2_t, power_feat_tot_t, theta_delta_t, emg_integrated_amp_t
del ratio_amplitude2_t, theta_paa_imputed2_t, theta1_paa_imputed2_t
del theta2_paa_imputed2_t, sigma_paa_imputed2_t, beta1_paa_imputed2_t, beta2_paa_imputed2_t
del gamma1_paa_imputed2_t, gamma2_paa_imputed2_t, delta_paa_imputed2_t, tot_paa_imputed2_t
del  power_feat_delta2_t, power_feat_theta2_t, power_feat_theta12_t, power_feat_theta22_t
del  power_feat_sigma2_t, power_feat_beta12_t, power_feat_beta22_t, power_feat_gamma12_t
del  power_feat_gamma22_t, power_feat_tot2_t, theta_delta2_t





sk1 = SelectKBest(k = 8)
sk1 = sk1.fit(X_train, y_train)
X_train_8best = sk1.transform(X_train)

search_model_params8b = GridSearchCV(
     SVC(kernel = 'rbf'),{
         'C': [0.8, 0.85, 0.9]
    }, cv=5, return_train_score = False, n_jobs = -1
)
search_model_params8b.fit(X_train_8best, y_train)

X_test = sc.transform(X_test)

X_test_sc1 = sk1.transform(X_test)
predictions = search_model_params8b.predict(X_test_sc1)
dfPredictions = pd.DataFrame(predictions)
dfPredictions.index.name = "id"
dfPredictions.to_csv("memsafe1task4.csv", header = ['y'], index=True)
del X_test_sc1


del X_train_8best

# ----------------------------------------------

sk2 = SelectKBest(k = 10)
sk2 = sk2.fit(X_train, y_train)
X_train_10best = sk2.transform(X_train)

search_model_params10b = GridSearchCV(
     SVC(kernel = 'rbf'),{
         'C': [0.8, 0.85, 0.9]
    }, cv=5, return_train_score = False, n_jobs = -1
)
search_model_params10b.fit(X_train_10best, y_train)
saveResults(search_model_params10b.best_params_, search_model_params10b.best_score_, 'svc10best')

X_test_sc2 = sk2.transform(X_test)
predictions = search_model_params10b.predict(X_test_sc2)
dfPredictions = pd.DataFrame(predictions)
dfPredictions.index.name = "id"
dfPredictions.to_csv("memsafe2task4.csv", header = ['y'], index=True)
del X_test_sc2

del X_train_10best

# ---------------------------------------------

sk3 = SelectKBest(k = 60)
sk3 = sk3.fit(X_train, y_train)
X_train_60best = sk3.transform(X_train)



search_model_params60b = GridSearchCV(
     SVC(kernel = 'rbf'),{
         'C': [0.8, 0.85, 0.9]
    }, cv=5, return_train_score = False, n_jobs = -1
)
search_model_params60b.fit(X_train_60best, y_train)


X_test_sc3 = sk3.transform(X_test)
predictions = search_model_params60b.predict(X_test_sc3)
dfPredictions = pd.DataFrame(predictions)
dfPredictions.index.name = "id"
dfPredictions.to_csv("memsafe3task4.csv", header = ['y'], index=True)
del X_test_sc3


del X_train_60best

# ---------------------------------------------


X_train_8_best = sk1.transform(X_train)



search_model_params_lda = GridSearchCV(
     LinearDiscriminantAnalysis(),{
       'n_components' : [2]
    }, cv=5, return_train_score = False, n_jobs = -1
)
search_model_params_lda.fit(X_train_8_best, y_train)

del X_train_8_best


# ****************************************************************



del y

del X_train
### predicting starts here

X_test_sc1 = sk1.transform(X_test)
predictions = search_model_params_lda.predict(X_test_sc1)
dfPredictions = pd.DataFrame(predictions)
dfPredictions.index.name = "id"
dfPredictions.to_csv("memsafe4task4.csv", header = ['y'], index=True)
del X_test_sc1


