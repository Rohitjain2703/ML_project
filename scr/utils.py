# in this file we write code realted to data base and clouds realated
import os
import sys  
import pandas as pd
import numpy as np
from scr.Exceptions import CustomException
import dill as pickle

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        #to create the directory if it does not exist

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            #to save the object in the file path
    except Exception as e:
        raise CustomException(e,sys)
        #to handle the exception and raise custom exception
