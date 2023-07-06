## Train model here
## Confusion matrix , All models are present here.
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging

from src.util import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

'''
A pickle file is a serialized object in Python, which means it is a binary representation of an object that can be stored on disk. 
The name "pickle" is derived from the process of preserving food by pickling, as pickling in Python refers to the process of preserving and storing objects.

Pickle files are created using the `pickle` module in Python, which provides functions for serializing (pickling) and deserializing (unpickling) objects.
Pickling allows you to convert complex Python objects, such as lists, dictionaries, classes, or even user-defined objects, into a compact binary format that can be easily stored,
transferred, and later reconstructed back into objects.

There are several reasons why we create pickle files:

1. Object Persistence: Pickle files are commonly used to save and restore objects in their exact state. 
   This is particularly useful when you want to save the state of an object or a model, such as a trained machine learning model, 
   for later use or sharing with others.

2. Data Serialization: Pickle files can be used to serialize data structures, such as lists or dictionaries, into a file. 
   This allows you to store complex data structures in a compact binary format and load them back into memory when needed.

3. Interprocess Communication: Pickle files can serve as a means of communication between different processes or systems. 
   Objects can be pickled in one process, written to a file, and then unpickled in another process, allowing for seamless data exchange.

4. Caching: Pickle files can be used for caching data or precomputed results. Instead of recomputing expensive operations,
   you can pickle the results and load them from the pickle file when needed, saving time and computational resources.
'''
class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()


    def initial_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Spliting traing and test data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )

            models = {
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }


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

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best Model found")
            
            logging.info("Model training dole and best model found")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)