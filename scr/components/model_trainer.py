# in ths file we train the ML model 
import os
import sys  
from scr.Exceptions import CustomException
from scr.logger import logging  
from dataclasses import dataclass
from scr.utils import save_object, evaluate_model


#models importing 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.metrics import r2_score  #this is used to evaluate the model performance



@dataclass
class modeltrainerconfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl') #this is the path where the trained model will be saved
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=modeltrainerconfig()
        #this will create an object of modeltrainerconfig class and we can access the path using this object

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")  #logging the splitting of data
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            #this will split the train and test array into input and target feature
            #all rows and all columns except last column will be input feature
            #all rows and only last column will be target feature

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            #this is a dictionary which contains the model name as key and model object as value


            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbours":{
                    'n_neighbors': [3,5,7,9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'p': [1,2]  #1=manhattan, 2=euclidean
                },



                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            #this will return a dictionary with model name as key and r2 score as value

            ## to get the best model score from the dictionary
            best_model_score=max(sorted(model_report.values()))
            #this will return the best r2 score from the dictionary

            ## to get the best model name from the dictionary
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            #this will return the best model name from the dictionary

            best_model=models[best_model_name]
            #this will return the best model object from the models dictionary

            if best_model_score<0.6:
                raise CustomException("No best model found")
            #if the best model score is less than 0.6 then we will raise an exception

            logging.info(f"Best model found on both training and testing dataset")
            #logging the best model name and score

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            #this will save the best model object in the specified path

            predicted=best_model.predict(X_test)
            #this will predict the target feature for the test data
            r2_square=r2_score(y_test,predicted)
            #this will calculate the r2 score for the test data predictions
            return r2_square
            #return the r2 score for the test data predictions

        except Exception as e:
            raise CustomException(e,sys)
            #to handle the exception and raise custom exception 

