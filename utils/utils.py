import os
import sys
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

import pickle
import mlflow
from utils.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train,X_test,y_test,models: dict,param: dict):
    try:
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        report = {}

        if mlflow.active_run():
            mlflow.end_run()
        print('Preparing for the evaluation of the models.')
        for model_name, model in models.items():
            # model = list(models.values())[i]
            para=param[model_name]
            
            with mlflow.start_run(run_name= model.__class__.__name__):
                gs = GridSearchCV(model,para,cv=2, scoring= 'accuracy', n_jobs=-1)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                #model.fit(X_train,y_train)
                mlflow.log_params(gs.best_params_)
                y_test_pred = gs.best_estimator_.predict(X_test)
                accuracy = accuracy_score(y_test, y_test_pred)
                f1 = f1_score(y_test, y_test_pred)
                
                mlflow.log_metrics({'accuracy': accuracy,'f1-score': f1})
                
                
                test_model_score = accuracy_score(y_test, y_test_pred)

                report[model_name] = test_model_score
                if model.__class__.__name__.startswith('RandomForestClassifier'):
                    mlflow.sklearn.log_model(
                        sk_model=gs.best_estimator_,
                        artifact_path="sklearn-model",
                        input_example=X_train,
                        registered_model_name= f"randomforest-model"
                    )
                else:
                    mlflow.xgboost.log_model(
                        gs.best_estimator_, 
                        artifact_path= 'xgboost-model',
                        registered_model_name= 'xgboost-model')

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)