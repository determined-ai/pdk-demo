import torch
import logging
import os
import json

import pandas as pd
from utils import scale_data, encode_categories, preprocess_dataframe

from ts.torch_handler.base_handler import BaseHandler
from torch.profiler import ProfilerActivity

logger = logging.getLogger(__name__)

class CustomerChurnHandler(BaseHandler):


    def __init__(self):
        super(CustomerChurnHandler, self).__init__()
        self.reference_df = pd.read_csv("reference_data.csv")
        
        object_cols = list(reference_df.columns[reference_df.dtypes.values == "object"])
        int_cols = list(reference_df.columns[reference_df.dtypes.values == "int"])
        float_cols = list(reference_df.columns[reference_df.dtypes.values == "float"])
        
        # Churn will be the label, no need to preprocess it
        int_cols.remove("churn")

        self.numerical_cols = int_cols+float_cols


    #def initialize(self, context): 
    def preprocess(self, requests):
        """Tokenize the input text using the suitable tokenizer and convert 
        it to tensor

        Args:
            requests: A list containing a dictionary, might be in the form
            of [{'body': json_file}] or [{'data': json_file}]
        """

        # unpack the data
        data = requests[0].get('body')
        if data is None:
            data = requests[0].get('data')
            
        df = pd.DataFrame.from_dict(data)
        logger.info('Successfully converted json/dict back to pandas DataFrame')                             
        
        preprocessed_df = preprocess_dataframe(df, self.reference_df, self.numerical_cols)
        logger.info('Dataframe successfully preprocessed')
        
        feature_cols = list(preprocessed_df.columns)
        label_col = "churn"
        feature_cols.remove(label_col)
        
        input_tensor = torch.Tensor(df[feature_cols].values)
        
        logger.info('Dataframe successfully converted to tensor')

        return input_tensor


    def inference(self, inputs):
    
        output = self.model(input_tensor)
        output[output < 0.5] = 0.0
        output[output >= 0.5] = 1.0
        logger.info('Predictions successfully created.')

        return predictions
    
    def postprocess(self, data):

        return data.tolist()
