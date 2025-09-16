# this file will read the data form vrious data base  or differnts data servers.
import pandas as pd
import os
import sys
from scr.Exceptions import CustomException
from scr.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from scr.components.data_transformation import DataTransformation #importing the data transformation class to use the data transformation config class
from scr.components.data_transformation import DataTransformationConfig #importing the data transformation config class to use the preprocessor object file path

from scr.components.model_trainer import modeltrainerconfig #importing the model trainer config class to use the trained model file path
from scr.components.model_trainer import ModelTrainer #importing the model trainer class to use the initiate model trainer method

@dataclass
class dataingestionconfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=dataingestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or compoents")
        try:
            df=pd.read_csv("notebook\data\stud.csv")
                 #read the data from the csv file/ database or any other source
            logging.info("Read the dataset as dataframe")

                 #in the nest logging we will save the data in the raw data folder


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
                # save the dataframes to the respective paths 


            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
                 # split the dataset into train and test save before spliting data 


            logging.info("Train test split initiated")
                #now we will split the data into train and test
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
                    # save the data in the respective paths then train data and test data in the specified paths


            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
                #train data will be saved in the train_data_path

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
                 #test data will be saved in the test_data_path


            logging.info("Ingestion of the data is completed")
                #logging say data ingestion is completed


            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                #return the train and test data path for further processing
            )

            
            


        except Exception as e:
                 #logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)
                #raise custom exception if any error occurs during the data ingestion


if __name__=="__main__": #this is the main function to call the data ingestion class
    obj=DataIngestion()  #create an object of the data ingestion class

    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_array, test_array ,preprocessor_path= data_transformation.initiate_data_transformation(train_data, test_data)
    
    #call the initiate_data_ingestion method to start the data ingestion process
    #this will run the data ingestion process and save the data in the artifacts folder

    ModelTrainer=ModelTrainer()
    print(ModelTrainer.initiate_model_trainer(train_array, test_array))