from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import multiprocessing as mp
import ctypes
import datetime
import time
import mne
import glob
import re
import numpy as np
import scipy
import scipy.stats
from sklearn.neighbors import NearestNeighbors
import random             
import matplotlib.pyplot as plt
import pywt
import networkx as nx

class Config():
  def __init__(self, data_path="./", patient=1):
    self.patient = patient
    self.patient_folder = 'chb{:02d}/'.format(patient)
    self.output_file = "./data/"+'chb{:02d}'.format(patient)+".dat"
    self.data_path = data_path
    #data parameters
    self.sampling=256 #Hz
    self.duration=5 # duration of samples to train
    self.preictal_duration=1800
    self.N_features=815
    
    self.channels_names=["FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3",
                         "C3-P3","P3-O1","FP2-F4","F4-C4","C4-P4","P4-O2",
                         "FP2-F8","F8-T8","T8-P8-0","P8-O2","FZ-CZ","CZ-PZ"]
    self.N_channels = len(self.channels_names)

class EEG_data():
  def __init__(self, eeg_signals, eeg_start, eeg_end, seizures_start, seizures_end, cfg):
    self.sampling = cfg.sampling
    self.duration = cfg.duration
    self.N_features = cfg.N_features
    self.preictal_duration = cfg.preictal_duration
    self.sample_size=cfg.sampling*cfg.duration
    self.eeg_signals = eeg_signals
    self.eeg_start = eeg_start
    self.eeg_end = eeg_end
    self.seizures_start = seizures_start
    self.seizures_end = seizures_end
    
    signal = self.get_signal(1000)
    self.get_max_correlation(signal)
    self.nonlinear_interdependence(signal)
    
  def get_signal(self,time):
    return self.eeg_signals.copy().crop(time,time + self.duration)
  
  def get_features(self, time):
    signal = self.get_signal(time)
    n_channels = len(signal)
    
    moments = self.compute_moments(signal)
    zero_crossings = self.get_zero_crossings(signal)
    peak_to_peak = self.get_peak_to_peak(signal)
    absolute_area = self.compute_absolute_area(signal)
    psd_ratio = self.power_spectral_density(signal)
    decorrelation_time = self.get_decorrelation_time(signal)
    dwt_coeffs = self.discrete_wavelet_transform(signal)
    max_correlation_coeffs = self.get_max_correlation(signal)
    
    EEG_graph = self.create_graph(max_correlation_coeffs)
    MST_graph = nx.minimum_spanning_tree(EEG_graph, weight='weight')
    graph_features = self.get_graph_features(MST_graph)
    
    features=moments.flatten() # 5xN
    features = np.hstack([features, np.triu(max_correlation_coeffs).flatten()]) # N(N-1)/2
    
    return features

  def get_max_correlation(self, signal):
    n_channels = len(signal.ch_names)
    corr_coeffs = np.identity( n_channels )
    auto_corr_coeffs = self.get_auto_corr_coeffs(signal);
    for channel_1 in range(0, n_channels-1):
      for channel_2 in range(channel_1+1, n_channels):
        corr_coeffs[channel_1,channel_2] = self.channel_correlation(signal,auto_corr_coeffs, channel_1, channel_2)
        corr_coeffs[channel_2,channel_1] = corr_coeffs[channel_1,channel_2]
    return corr_coeffs
  
  def get_auto_corr_coeffs(self, signal):
    n_channels = len(signal.ch_names)
    auto_corr_coeffs=np.zeros(n_channels)
    for channel in range(0,n_channels):
      auto_corr_coeffs[channel] = scipy.correlate(signal._data[channel], signal._data[channel],"valid")
    return auto_corr_coeffs
    
  def channel_correlation(self, signal, auto_corr_coeffs, channel_1, channel_2):
    signal_1 = signal._data[channel_1];
    signal_2 = signal._data[channel_2];
    corr_1 = auto_corr_coeffs[channel_1]
    corr_2 = auto_corr_coeffs[channel_2]
    corr = scipy.correlate(signal_1, signal_2,"full")
    max_corr = np.max(corr)
    max_corr = max_corr / np.sqrt(corr_1*corr_2)
    return max_corr
  
  def nonlinear_interdependence(self, signal):
    n_channels = len(signal.ch_names)
    embedded_signals = self.embed_signals(signal)
    full_S = self.get_full_S(embedded_signals)
    S = np.empty( int(n_channels * (n_channels-1) /2))
    idx=0
    for channel_1 in range(0, n_channels):
      for channel_2 in range(channel_1+1, n_channels):
        S[idx] = (full_S[channel_1, channel_2] + full_S[channel_2, channel_1]) / 2
        idx+=1
    print(S)
    return S
    
  def embed_signals(self,signal,d=10,lag=6):
    n_channels = len(signal.ch_names)
    len_signal =  len(signal._data[0])
    embedded_signals = np.empty([n_channels,len_signal-(d-1)*lag,d])
    for channel in range(0, n_channels):
      for i in range(0, len_signal- (d-1)*lag) :
        for j in range(0,d):
          idx = i - j*lag + (d-1)*lag
          embedded_signals[channel, i,j] = signal._data[channel][idx]
    return embedded_signals
    
  def get_full_S(self, embedded_signals):
    n_channels = len(embedded_signals)
    full_S=np.empty([n_channels, n_channels])
    for channel_1 in range(0, n_channels):
      for channel_2 in range(0, n_channels):
        if(channel_1 != channel_2):
          full_S[channel_1,channel_2] = self.get_S(embedded_signals[channel_1,:,:],embedded_signals[channel_2,:,:])
    return full_S
  
  def get_S(self, xa, xb): 
    K = 5
    nbrs_a = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree', metric='euclidean').fit(xa)
    nbrs_b = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree', metric='euclidean').fit(xb)
    distances_a, indices_a = nbrs_a.kneighbors(xa)
    distances_b, indices_b = nbrs_b.kneighbors(xb)

    S=0
    for t in range(0,len(xa)):
      ra=np.average(np.square(distances_a[t,1:]))
      rab=0
      for k in range(0,K):
        rab+=np.sum(np.square(xa[t]-xa[indices_b[t][k+1]]))
      rab=rab/(K)
      S+=ra/rab
    S=S/len(xa)
    return S
    
  def compute_DSTL(self, signal):
    n_channels = len(embedded_signals)
    embedded_signals = self.embed_signals(signal,d=7,lag=6)
    N = len(embedded_signals[0])
    for channel in range(0,n_channels):
      for i in range(0,N):
        self.get_perturbation(embedded_signals[channel], i)
    
  def get_perturbations(self, embedded_signal, i):
    j = self.get_transverse(embedded_signal,i)
      
  def get_transverse(self, embedded_signal, i):
    
   
class Patient_data():
  
  def __init__(self, cfg):
    self.data_path = cfg.data_path
    self.patient_folder = cfg.patient_folder
    self.output_file = cfg.output_file
    self.duration = cfg.duration
    self.sampling = cfg.sampling
    self.channels_names = cfg.channels_names
    self.eeg_data={}
    
    print(self.data_path+self.patient_folder)
    summary_filename = glob.glob(self.data_path+self.patient_folder+"/*summary.txt")[0]
    self.seizures_time=[]
    self.load_summary(summary_filename)
    self.current_time = 0
    self.current_day = 0
    self.load_files(cfg)
    self.create_segments()
    self.save_segments()
 
  def save_segments(self):
    f = open(self.output_file, 'w')
    self.write_seizures(f)
    for key in self.eeg_data:
      segments=self.eeg_data[key].get_all_segments_par()
      self.write_segments(segments, f)
  
  def write_segments(self, segments, f):
    for segment in segments:
      f.write(" ".join(str(x) for x in segment))
      f.write('\n')
    
  def write_seizures(self, f):
    for st in self.seizures_time:
      f.write(str(st[0])+' '+str(st[1])+' ')
    f.write('\n')
    
  def create_segments(self):
    for key in self.eeg_data:
      self.eeg_data[key].segment_signals()
    # TODO rewrite labeling part...
    for key in self.eeg_data:
      for s in  range(0,len(self.eeg_data[key].seizures_start)):
        next_seizure_time_start = self.eeg_data[key].seizures_start[s]
        next_seizure_time_end = self.eeg_data[key].seizures_end[s]
        for key_2 in self.eeg_data:
          self.eeg_data[key_2].label_segments(next_seizure_time_start, next_seizure_time_end)
          
  def load_summary(self, summary_filename):
    f=open(summary_filename)
    self.summary=f.readlines()
    f.close()
    
  def load_files(self, cfg):
    line_to_match="File Name:"
    indices = [index for index,line in enumerate(self.summary) if line_to_match in line]
    # ~ for idx in indices:
    for idx in indices[0:2]:
      filename = re.search('File Name: (.+?)\n', self.summary[idx] ).group(1)
      file_number = re.search('_(.+?).edf', self.summary[idx]).group(1)
      print("Loading: " + filename)
      self.eeg_data[file_number] = self.load_data(self.data_path+self.patient_folder+filename, idx, cfg)
    
  def load_data(self, filename, index_begin, cfg):
    # get info in the summary file
    file_number = re.search('_(.+?).edf', self.summary[index_begin]).group(1)
    
    eeg_start = re.search('Time: (.*?)\n', self.summary[index_begin+1]).group(1)
    if(eeg_start[0:2]=='24'):
      eeg_start=str.replace(eeg_start,'24','00')
    eeg_start = time.strptime(eeg_start,'%H:%M:%S')
    eeg_start = datetime.timedelta(hours=eeg_start.tm_hour,minutes=eeg_start.tm_min,seconds=eeg_start.tm_sec).total_seconds()
    eeg_end = re.search('Time: (.*?)\n', self.summary[index_begin+2]).group(1)
    if(eeg_end[0:2]=='24'):
      eeg_end=str.replace(eeg_end,'24','00')
    eeg_end = time.strptime(eeg_end,'%H:%M:%S')
    eeg_end = datetime.timedelta(hours=eeg_end.tm_hour,minutes=eeg_end.tm_min,seconds=eeg_end.tm_sec).total_seconds()
    
    # next day
    if(eeg_start<self.current_time):
      self.current_day=self.current_day+1
    self.current_time = eeg_start
    eeg_start = int(eeg_start + 24*3600*self.current_day)
    eeg_end = int(eeg_end + 24*3600*self.current_day)
    n_seizures =  int(re.search(':(.+?)\n', self.summary[index_begin+3]).group(1))
    seizures_start = [0]*n_seizures
    seizures_end = [0]*n_seizures
    # load the seizures info in the summary file
    for i in range(0,n_seizures):
      idx=index_begin+4+2*i
      seizures_start[i] = int(re.search(':(.+?) seconds\n', self.summary[idx]).group(1))+eeg_start
      seizures_end[i] = int(re.search(':(.+?) seconds\n', self.summary[idx+1]).group(1))+eeg_start
      self.seizures_time.append( [ seizures_start[i], seizures_end[i] ])
      
    eeg_signals=mne.io.read_raw_edf(filename, preload=True, stim_channel=None)
    eeg_signals.pick_channels(self.channels_names)
    # return the EEG signal
    return EEG_data(eeg_signals, eeg_start, eeg_end, seizures_start, seizures_end, cfg)
  
  def print_info(self):
    print("Number of files loaded: " + str(len(self.eeg_data)) )


cfg = Config(data_path ="/home/sichhadmin/epilepsy/data/CHB-MIT/" , patient = 1)
patient_data = Patient_data(cfg)
