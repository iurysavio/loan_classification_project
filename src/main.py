import mlflow
import os
import sys

from utils.logger import logging
from utils.exception import CustomException

from src.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

try:
    mlflow.set_tracking_uri('http://127.0.0.1:5000')

    with mlflow.start_run() as run:
        print('Starting data ingestion\n')
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        print('Starting data transformation\n')
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        print('Starting model training\n')
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))
except Exception as e:
    raise CustomException(e, sys) 