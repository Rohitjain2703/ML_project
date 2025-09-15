# here we do data transfromation all EDA in this file.
import sys
import os     
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

#to handle the exception and handle the logging and data class for configuration
from dataclasses import dataclass   
from scr.Exceptions import CustomException
from scr.logger import logging  

from scr.utils import save_object 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts','preprocessor.pkl')
        
        #this is the path where we will save the preprocessor object after creating it.


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
        #initialize the data transformation config class to access the preprocessor object file path.


        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''


        try:
            numerical_columns=['writing_score','reading_score']
            #numerical columns are writing score and reading score

            categorical_columns=['gender','race_ethnicity',
                                 'parental_level_of_education',
                                 'lunch',
                                 'test_preparation_course']
            
            #categorical columns are gender, race/ethnicity, parental level of education, lunch, test preparation course
            # we will do scaling and encoding of the categorical and numerical columns

            numerical_columns_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler(with_mean=False))
            ])


            #for numerical columns we will use median imputer and standard scaler
            #imputer will fill the missing values with median value of the column
            #scaler will scale the values to standard normal distribution
            #pipeline is a sequence of data processing steps
            #missing values are filled with median value of the column




            categorical_columns_pipeline=Pipeline(
                steps=[
                    ("OneHotEncoder",OneHotEncoder(handle_unknown='ignore')), #for categorical columns we will use one hot encoder and standard scaler
                    ('imputer',SimpleImputer(strategy='most_frequent')), #imputer will fill the missing values with most frequent value of the column
                    ('scaler',StandardScaler(with_mean=False)) #scaler will scale the values to standard normal distribution
                    
                ]
            )
            logging.info(f"numerical columns {numerical_columns}")
            #logging info for numerical columns encoding and scaling completed

            logging.info(f"Categorical columns {categorical_columns}")
            #logging info for categorical columns encoding completed
            
            preprocessor=ColumnTransformer( #column transformer is used to apply different transformations to different columns
                #we will apply the numerical columns pipeline to the numerical columns and categorical columns pipeline to the categorical columns
                [
                    ('numerical_columns_pipeline',numerical_columns_pipeline,numerical_columns),
                    ('categorical_columns_pipeline',categorical_columns_pipeline,categorical_columns)
                ]
            )

            return preprocessor
            #return the preprocessor object
        except Exception as e:
            raise CustomException(e, sys)
            #raise custom exception if any error occurs during the data transformation process  





    def initiate_data_transformation(self,train_path,test_path):
        try:
            #reading the train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame head : \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame head : \n{test_df.head().to_string()}")
            
            #get the preprocessor object
            preprocessor_obj=self.get_data_transformer_object()
            target_column_name='math_score'
            numerical_columns=['writing_score','reading_score']
            
            #separating input features and target feature from training and testing dataframe
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Separated input features and target feature from training and testing dataframe")
            
            #transforming the input features using the preprocessor object
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            logging.info("Applied preprocessing object on training and testing dataframe")
            
            #concatenating the input features and target feature to create the final train and test arrays
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info("Concatenated input features and target feature to create the final train and test arrays")
            
            #saving the preprocessor object to the file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            logging.info("Saved the preprocessor object to the file")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
            #raise custom exception if any error occurs during the data transformation process  