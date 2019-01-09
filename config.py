import numpy as np

class Config():
  def __init__(self, data_path="./", NN="TCN", patient=1):
    self.patient = patient
    # ~ self.features=["max_correlation","SPLV","nonlinear_interdependence","DSTL"]
    self.features=["max_correlation","DSTL","nonlinear_interdependence"]
    self.input_files=[]
    for feature in self.features:
      self.input_files.append(data_path + '/' + 'chb{:02d}'.format(patient) +'/'+feature+'.dat')
    self.NN = NN
    # ~ self.segments_type_train = "continuous"
    # ~ self.segments_type_train = "random"
    # ~ self.segments_type_test = "continuous"
    # ~ self.segments_type_test = "random"
    # ~ self.segments_type = "continuous"
    self.data_path = data_path
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
