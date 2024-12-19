import os
import sys

import pandas as pd
import numpy as np 

from utils.logger import logging
from utils.exception import CustomException
from utils.utils import save_object

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engineering import create_features

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for the data transformation
        '''
        try:
            # Setting the type of the columns in the dataset
            numerical_columns = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                                 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                                 'credit_score', 'loan_income_ratio', 'weighted_credit_history']
            categorical_columns = ['person_gender', 'person_education', 'person_home_ownership',
                                   'loan_intent', 'previous_loan_defaults_on_file', 'age_range',
                                   'credit_score_category', 'emp_exp_group', 'income_category']

            # Creating the pipelines for the categorical and numerical

            num_pipeline = Pipeline(
                steps=[
                   ('scaler', StandardScaler(with_mean= False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean= False))
                ]
            )
            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info('Categorical and numeric transformation pipelines created')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            logging.info('Preprocessor pipeline object created')

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Train and test set loaded')
            logging.info('Obtaining preprocessor object')

            train_df = create_features(train_df)
            test_df = create_features(test_df)

            logging.info('New features created')
            
            target_column_name= 'loan_status'

            preprocessor_obj = self.get_data_transformer_object()
            
            input_feature_train_df= train_df.drop(columns=target_column_name, axis= 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns=target_column_name, axis= 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f'Applying preprocessor object on training and test dataframes')

            input_features_train_transformed = preprocessor_obj.fit_transform(input_feature_train_df)
            input_features_test_transformed = preprocessor_obj.transform(input_feature_test_df)

            train_array = np.c_[input_features_train_transformed, np.array(target_feature_train_df)]
            test_array = np.c_[input_features_test_transformed, np.array(target_feature_test_df)]

            logging.info('Saving preprocessing object')

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessor_obj
            )
            
            logging.info('Preprocessor object saved.')

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)