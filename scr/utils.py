# in this file we write code realted to data base and clouds realated
import os
import sys  
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from scr.Exceptions import CustomException
import dill as pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]

            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)


            y_train_pred = model.predict(X_train)








            #this will give the model object
            #.fit(X_train,y_train)
            #fit the model
            y_test_pred=model.predict(X_test)
            #predict the train data
            train_model_score = r2_score(y_train, y_train_pred)
            #predict the test data
            test_model_score=r2_score(y_test,y_test_pred)

            #calculate the r2 score
            report[list(models.keys())[i]]=test_model_score
            #this will give the model name and its score in the report dictionary

        return report
        #return the report dictionary

    except Exception as e:
        raise CustomException(e,sys)
        #to handle the exception and raise custom exception
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
            #to load the object from the file path
    except Exception as e:
        raise CustomException(e,sys)
        #to handle the exception and raise custom exception     