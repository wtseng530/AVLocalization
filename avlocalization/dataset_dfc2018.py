import numpy as np
import torch

class DatasetDFC2028:

    def __init__ (self, config): 

        self.config = config

        # load the data

    def __len__(self):
        return self.config.num_data_per_epoch


    def __getitem__(self, index):

        # create the dictionnary

        data = {}
        data["points"]=None
        data["image"]=None

        return data