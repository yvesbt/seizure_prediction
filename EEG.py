from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import multiprocessing as mp
import ctypes
import datetime
import time
import bisect
import numpy as np
import scipy
import scipy.stats
import random             
import matplotlib.pyplot as plt
import pywt
import networkx as nx

random.seed(0)

class Patient_data():
  '''
    Class Patient_data loads data in the following format:
          seizure_1_start seizure_1_end seizure_2_start ...
          time_1 feature_1 feature_2 feature_3 ...
          time_2 feature_1 feature_2 feature_3 ...
    seizures_times are stored in seizures_start and seizures_end
    features are stored in segments (without time)
    times are stored separately to feature in times
  '''
  def __init__(self, cfg):
    self.data_path = cfg.data_path
    self.segments_type_train = cfg.segments_type_train
    self.segments_type_test = cfg.segments_type_test
    self.preictal_duration = cfg.preictal_duration
    self.timesteps = cfg.timesteps
    self.num_input = cfg.num_input
    self.num_classes = cfg.num_classes
    
    self.segments, self.times, self.seizures_start, self.seizures_end = self.load_data(cfg)
    self.N_seizures = len(self.seizures_start)
    self.annotations = self.annotate_data();
    # ~ self.leave_one_seizure(5)
    self.k_fold_division(2)
    
  def annotate_data(self):
    annotations = np.zeros(len(self.times),dtype=int)
    for seizure_start in self.seizures_start:
      for i in range(0, len(self.times) ):
        if( seizure_start > self.times[i] and seizure_start - self.times[i] < self.preictal_duration):
          annotations[i]=1
          
    for s in range(0,len(self.seizures_start)):
      for i in range(0, len(self.times) ):
        if( self.seizures_start[s] < self.times[i] and self.seizures_end[s] > self.times[i]):
          annotations[i]=2
    return annotations
    
    
  def k_fold_division(self, k):
    '''
       input : k to divide the set into k parts 
    '''
    assign_to_set = np.zeros(len(self.annotations))
    
    indices = np.arange(0,len(self.segments))
    # ~ np.random.shuffle(indices)
    for i in range(0,int(len(indices)*(1-1./k))):
      idx = indices[i]
      if(self.annotations[idx] == 0):
        assign_to_set[idx] = 1
      elif(self.annotations[idx] == 1):
        assign_to_set[idx] = 2
    for i in range(int(len(indices)*(1-1./k)),len(indices)):
      idx = indices[i]
      if(self.annotations[idx] == 0):
        assign_to_set[idx] = 3
      elif(self.annotations[idx] == 1):
        assign_to_set[idx] = 4
    self.assign_sets(assign_to_set)
    
  def leave_one_seizure(self, i):
    '''
      input : i for seizure to leave out
      This is used for Cross-Validation and leave 1 or the seizures
      out for cross-validation. Training set are indices of segments,
      same for validation set. Apart from seizure, a set without
      seizure is added (following the seizure)
    '''
    seizure_start = self.seizures_start[i]
    seizure_end = self.seizures_end[i]
    i_start = bisect.bisect_left(self.times, seizure_start)
    i_end = bisect.bisect_left(self.times, seizure_end)
    self.divide_data(seizure_start, seizure_end)
    
  def divide_data(self, seizure_start, seizure_end):
    '''
      input : seizure_start and seizure_end for seizure to keep for test set
      Make sure that all indices are in the preictal state before adding to
      test with seizure (test_ws)
      assign_to_set : 1 for train_wos, 2 train_ws, 3 test_wos, 4 test_ws
    '''
    assign_to_set = np.zeros(len(self.annotations))
     
    test_start_ws = bisect.bisect_left(self.times, seizure_start - self.preictal_duration)
    test_end_ws = bisect.bisect_left(self.times, seizure_start)
    length_ws = 0
    for idx in range(test_start_ws,test_end_ws):
      if(self.annotations[idx] == 1):
        assign_to_set[idx] = 4
        length_ws+=1
    idx = test_start_wos = bisect.bisect_right(self.times, seizure_end)
    length_wos=0
    while( length_wos < length_ws and idx<len(self.annotations) ):
      if( self.annotations[idx] == 0):
        assign_to_set[idx] = 3
        length_wos+=1
      idx+=1
    
    for i in range(0, len(self.annotations)):
      if(assign_to_set[i] == 0):
        if( self.annotations[i] == 0 ):
          assign_to_set[i] = 1
        if( self.annotations[i] == 1 ):
          assign_to_set[i] = 2
    self.assign_sets(assign_to_set)
    
  def assign_sets(self,assign_to_set):
    '''
      input : assign_to_set with len(segments) indices from 1 to 4
      assign_to_set : 1 for train_wos, 2 train_ws, 3 test_wos, 4 test_ws
    '''
    self.train_wos = []
    self.train_ws = []
    self.test_wos = []
    self.test_ws = []
    for idx in range(0,len(assign_to_set)):
      if(assign_to_set[idx]==1):
        self.train_wos.append(idx)
      if(assign_to_set[idx]==2):
        self.train_ws.append(idx)
      if(assign_to_set[idx]==3):
        self.test_wos.append(idx)
      if(assign_to_set[idx]==4):
        self.test_ws.append(idx)

  def train_next_batch(self,batch_size, num_input):
    train_batch = np.empty( [batch_size, num_input, self.N_features] )
    train_annotations = np.empty( [batch_size,2] )
    for i in range(0,batch_size):
      p = random.random()
      if(p<0.5):
        train_batch[i] = self.get_batch(self.train_wos, num_input, self.segments_type_train)
        train_annotations[i] = [1,0]
      else:
        train_batch[i] = self.get_batch(self.train_ws, num_input, self.segments_type_train)
        train_annotations[i] = [0,1]
    return train_batch, train_annotations
  
  def get_test_batch(self,batch_size, num_input):
    test_batch = np.empty( [batch_size, num_input, self.N_features] )
    test_annotations = np.empty( [batch_size,2] )
    for i in range(0,batch_size):
      p = random.random()
      if(p<0.5):
        test_batch[i] = self.get_batch(self.test_wos, num_input, self.segments_type_test)
        test_annotations[i] = [1,0]
      else:
        test_batch[i] = self.get_batch(self.test_ws, num_input, self.segments_type_test)
        test_annotations[i] = [0,1]
    return test_batch, test_annotations
    
  def get_batch(self, idx_set, num_input, data_type):
    batch = np.empty([ num_input,self.N_features])
    if(data_type == "random"):
      batch_indices = random.sample(idx_set, num_input)
    elif(data_type == "continuous"):
      data_continuous = False
      while(not data_continuous):
        idx = random.randrange( 0, len(idx_set) - num_input )
        data_continuous = self.check_continuous(idx_set, idx, num_input)
      batch_indices=range(idx, idx+num_input)
    for i in range(0,num_input):
      batch[i,:] = self.segments[ batch_indices[i] ]
    return batch
  
  def check_continuous(self, idx_set, idx, num_input):
    if(idx_set[idx] + num_input == idx_set[idx+num_input]):
      return True
    else:
      return False
    
  def load_data(self, cfg):
    f=open(cfg.input_file,'r')
    line = f.readline()
    seizures_start=[]
    seizures_end=[]
    seizures_time = [int(t) for t in line.strip().split(' ')];
    for i in range(0, int(len(seizures_time)/2)):
      seizures_start.append( seizures_time[2*i] )
      seizures_end.append( seizures_time[2*i+1] )
    
    lines = [line.rstrip().split(' ') for line in f]
    self.N_features = len(lines[0])-1
    segments=np.empty([len(lines),self.N_features+1])

    segments = np.float32( lines )
    times = segments[:,0]
    segments = segments[:,1:]
    return segments,times, seizures_start, seizures_end
      
