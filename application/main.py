import os
import sys
import pandas as pd
import pickle

import mlflow
from mlflow.tracking import MlflowClient # type: ignore

from pydantic import BaseModel
from fastapi import FastAPI

from utils.utils import load_object
from utils.feature_engineering import create_features, create_features_for_inference
from utils.logger import logging
from utils.exception import CustomException

mlflow.set_tracking_uri('http://127.0.0.1:5000')
client = MlflowClient()
model_name = "xgboost-model"
model_version = "latest"
model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.xgboost.load_model(model_uri)

class BodyModel(BaseModel):
    person_age: int
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: int
    person_home_ownership: str 
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int
    previous_loan_defaults_on_file: str

app = FastAPI()

@app.post('/predict')
async def predict(body: BodyModel):
    try:
        data = body.model_dump()
        data = pd.json_normalize(data)
        preprocessor_path = '/home/icavalca/workspace/repo/loan_classification_project/artifacts/preprocessor.pkl'
        preprocessor = load_object(preprocessor_path)
        data = create_features_for_inference(data)
        data_prepared = preprocessor.transform(data)
        prediction = model.predict(data_prepared)

        # Caso `prediction` seja um valor escalar, retorne diretamente
        if isinstance(prediction, (int, float)):
            prediction_value = prediction
        else:
            # Caso seja um array ou lista, pegue o primeiro valor
            prediction_value = int(prediction[0])

        return {'prediction': prediction_value}
    except Exception as e:
        raise CustomException(e, sys)

   
# preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
# preprocessor = load_object(preprocessor_path)
# print(preprocessor_path)
# print(type(preprocessor))