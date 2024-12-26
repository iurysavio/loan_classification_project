import os
import sys
from dataclasses import dataclass

from utils.logger import logging
from utils.exception import CustomException
from utils.utils import evaluate_models, save_object

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri('http://127.0.0.1:5000')
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
        
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                # "Decision Tree": DecisionTreeClassifier(),
                "XGBoost": XGBClassifier()
            }
            params = {
                'Random Forest': {
                        'n_estimators' : [50, 100, 200],
                        'max_depth' : [5, 10],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                 },
                'XGBoost': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6],
                        'learning_rate': [0.01, 0.1],
                        'min_child_weight': [1, 5]
                } 
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                        models=models,param=params)
            best_model_score = max(sorted(model_report.values()))

            # ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                 list(model_report.values()).index(best_model_score)
             ]
            return print(f'O melhor modelo foi o {best_model_name} que teve uma acurácia de {best_model_score:.2f}')
            # best_model = models[best_model_name]
            
            # if best_model_score<0.6:
            #     raise CustomException("No best model found")
            # logging.info(f"Best found model on both training and testing dataset")

            # save_object(
            #     file_path=self.model_trainer_config.trained_model_file_path,
            #     obj=best_model
            # )

            # predicted=best_model.predict(X_test)
            # accuracy = accuracy_score(y_test, predicted)
                           
        except Exception as e:
            raise CustomException(e, sys)