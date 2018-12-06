class Config():
  def __init__(self, data_path="./", NN="TCN", patient=1):
    self.patient = patient
    self.input_file = data_path + '/' + 'chb{:02d}'.format(patient) +'.dat'
    self.NN = NN
    # ~ self.segments_type_train = "continuous"
    self.segments_type_train = "random"
    # ~ self.segments_type_test = "continuous"
    self.segments_type_test = "random"
    self.segments_type = "continuous"
    self.data_path = data_path
    #data parameters
    self.sampling=256 #Hz
    self.duration=5 # duration of samples to train
    self.test_len = 250
    self.preictal_duration=1800
    # Training Parameters
    self.learning_rate = 0.001
    self.training_steps = 30
    self.batch_size = 32
    self.display_step = 1

    # Network Parameters
    self.num_input = 20 # data input (# signals)
    # ~ timesteps = 1155 # timesteps
    self.timesteps = 815 # timesteps
    self.num_hidden = 256 # hidden layer num of features
    self.levels = 6
    self.num_classes = 2 # Two classes: Inter and pre
    
    self.channels_names=["FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3",
                         "C3-P3","P3-O1","FP2-F4","F4-C4","C4-P4","P4-O2",
                         "FP2-F8","F8-T8","T8-P8-0","P8-O2","FZ-CZ","CZ-PZ"]
    self.N_channels = len(self.channels_names)

    

