##Data is transformed here i.e. Change Categorical to numerical etc.
import os
import sys
from src.exceptions import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.util import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')#If I want to create any models and want to save it into a pickel file. To Do so we need a single file path.


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This Function is resonponsible for data transformation.
        '''

        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]
            '''
            The Pipeline class in sklearn provides a simple and efficient way to define and manage pipelines.
            It encapsulates a series of transformations (preprocessing) and an estimator (modeling) into a single object.
            Each step in the pipeline is defined as a tuple of a string name and an instance of a transformer or an estimator.

            The pipeline allows you to conveniently chain together multiple data preprocessing and modeling steps,
            ensuring that the data flows smoothly through each step.
            '''
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns handeled")


            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading of Train and Test data completed")

            logging.info("Obtaing preprocessing object")

            preprocesing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['writing_score','reading_score']
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis = 1)
            traget_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis = 1)
            traget_feature_test_df = test_df[target_column_name]

            logging.info(" Applying preprocessing object on training and testing Dataframe")

            input_feature_train_arr = preprocesing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocesing_obj.fit_transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_arr,np.array(traget_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(traget_feature_test_df)]

            logging.info(f"Saving preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocesing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            