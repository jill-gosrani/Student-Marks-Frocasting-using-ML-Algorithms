## Functionality that are common and used in entire applications are written here 
# Example : Create a mongodb client to read a dataset, Save model to cloud, etc.

import os
import sys
from src.exceptions import CustomException
import dill

import numpy as np
import pandas as pd


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)