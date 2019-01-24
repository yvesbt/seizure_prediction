import numpy as np
'''
  The config class defines the the parameters for the handling of data and training. This input file 
  was created for patient 1 of the CHB-MIT, and should probably be adapted for other patients datasets.
  For instance, the number of channels could change. A text file could store the information for specific
  dataset and be used to define the relevant files and channels.
  
  data_path: path to the folder with features files
  self.patient: number of the patient (in CHB-MIT the patient are numbered)
  self.features: which features to use, corresponding to file in data_path
  self.NN: the type of neural network
  self.segments_type_train: type of segments for training (random, continuous, pattern)
  self.segments_type_test: type of segments for testing (random, continuous, pattern)
  self.sampling: sampling rate of the inputs files (in Hz)
  self.duration: length of segments of the input files
  self.test_len: number of sequences used for testing
  self.preictal_duration: time in seconds at which we define segments to be in the preictal period
  self.learning_rate: learning rate of the optimizer
  self.training_steps: number of steps used for the training of the neural network
  self.batch_size: number of sequences used for training
  self.display_step: display loss and accuracy every n steps
  self.num_inputs: number of segments put together to make a sequence
  self.num_hidden: number of neurons in the hidden layers (for LSTM and FC neural networks)
  self.levels: depth of the neural network (TCN only)
  self.num_classes: number of classes (usually preictal and interictal)
  self.channels_names: names of channels used (it should correspond to the ones used to create the features)
  self.selected_channels: choose which channel will be used. Only the selected channels will be loaded
  self.m_features: number of 'rows' for each feature. It is usually 1 but if the features has more than
                    one frequency, this number equals the number of frequencies
  self.N_features: number of features, this number is equal to the sum of m_features times N_channels*(N_channels-1)/2
'''
class Config():
  def __init__(self, data_path="./", NN="TCN", patient=1):
    self.data_path = data_path
    self.patient = patient
    # ~ self.features=["max_correlation","SPLV","nonlinear_interdependence","DSTL"]
    self.features=["max_correlation","DSTL","nonlinear_interdependence"]
    self.input_files=[]
    for feature in self.features:
      self.input_files.append(self.data_path + '/' + 'chb{:02d}'.format(patient) +'/'+feature+'.dat')
    self.NN = NN
    self.segments_type_train = "continuous"
    # ~ self.segments_type_train = "random"
    self.segments_type_test = "continuous"
    # ~ self.segments_type_test = "random"
    # ~ self.segments_type = "continuous"
    
    #data parameters
    self.sampling=256 #Hz
    self.duration=5 # duration of samples to train
    self.test_len = 250
    self.preictal_duration=1800
    # Training Parameters
    self.learning_rate = 0.01
    self.training_steps = 2000
    self.batch_size = 32
    self.display_step = 1

    # Network Parameters
   
    # ~ timesteps = 1155 # timesteps
    self.num_inputs = 60 # timesteps
    self.num_hidden = 256 # hidden layer num of features
    self.levels = 6
    self.num_classes = 2 # Two classes: Inter and pre
    
    self.channels_names=["FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3",
                         "C3-P3","P3-O1","FP2-F4","F4-C4","C4-P4","P4-O2",
                         "FP2-F8","F8-T8","T8-P8-0","P8-O2","FZ-CZ","CZ-PZ"]
    self.selected_channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    self.m_features = 0
    self.m_feature={}
    self.m_feature["max_correlation"] = 1
    self.m_feature["SPLV"] = 7
    self.m_feature["nonlinear_interdependence"] = 1
    self.m_feature["DSTL"] = 1
    
    for feature in self.features:
      self.m_features+=self.m_feature[feature]
      
    self.N_tot_channels = len(self.channels_names)
    self.N_channels = len(self.selected_channels)
    self.N_tot_features = int(self.N_tot_channels * (self.N_tot_channels-1)/2.) * self.m_features # data input (# signals)
    self.N_features = int(self.N_channels * (self.N_channels-1)/2.) * self.m_features # data input (# signals)
