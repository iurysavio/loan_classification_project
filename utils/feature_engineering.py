import pandas as pd
import numpy as np

def create_features(df):
    new_df = df.copy()
    new_df = new_df[new_df['person_emp_exp'] <= 76]
    # Implementing the new features:

    ## Loan Income Ratio
    new_df['loan_income_ratio'] = new_df['loan_amnt'] / new_df['person_income']

    ## Credit Score Category
    new_df['credit_score_category'] = pd.cut(
        new_df['credit_score']
        , bins= [300, 579, 669, 739, 799, 850]
        , labels= ['Very poor', 'Fair', 'Good', 'Very good', 'Exceptional']
    )

    ## Employment Experience Groups
    new_df['emp_exp_group'] = pd.cut(
        new_df['person_emp_exp']
        , bins= [-1, 2, 5, 10, float('inf')]
        , labels=['Entry-level', 'Mid-career', 'Experienced', 'Veteran']
    )

    ## Weighted Credit History Score
    new_df['weighted_credit_history'] = new_df['credit_score']*new_df['cb_person_cred_hist_length']

    ## Income category
    new_df['income_category'] = pd.qcut(
        new_df['person_income']
        , q= 4
        , labels=['Low', 'Lower-Middle', 'Upper-Middle', 'High']
    )

    
    new_df['age_range'] = new_df['person_age'].apply(lambda x : '20-30' if x < 30
                                                else '30-40' if 30 <= x < 40
                                                else '40-60' if 40 <= x < 60
                                                else '> 60')

    ## Changing the types of categorical columns
    for c in new_df.select_dtypes(include='object'):
        new_df[c] = new_df[c].astype('category')
    
    return new_df

import pandas as pd
import numpy as np
import pickle
import os

def create_features_and_save_metadata(df, output_dir="artifacts"):
    """
    Cria features no dataframe e salva metadados (bins, limites) necessários para inferência.

    :param df: DataFrame original.
    :param output_dir: Diretório onde os arquivos .pkl serão salvos.
    :return: DataFrame com as novas features.
    """
    os.makedirs(output_dir, exist_ok=True)  # Garante que o diretório exista

    new_df = df.copy()
    new_df = new_df[new_df['person_emp_exp'] <= 76]

    # Implementando as novas features:

    ## Loan Income Ratio
    new_df['loan_income_ratio'] = new_df['loan_amnt'] / new_df['person_income']

    ## Credit Score Category
    credit_score_bins = [300, 579, 669, 739, 799, 900]
    credit_score_labels = ['Very poor', 'Fair', 'Good', 'Very good', 'Exceptional']
    new_df['credit_score_category'] = pd.cut(
        new_df['credit_score'],
        bins=credit_score_bins,
        labels=credit_score_labels
    )

    # Salva os bins para uso na inferência
    with open(os.path.join(output_dir, 'credit_score_bins.pkl'), 'wb') as f:
        pickle.dump({'bins': credit_score_bins, 'labels': credit_score_labels}, f)

    ## Employment Experience Groups
    emp_exp_bins = [-1, 2, 5, 10, float('inf')]
    emp_exp_labels = ['Entry-level', 'Mid-career', 'Experienced', 'Veteran']
    new_df['emp_exp_group'] = pd.cut(
        new_df['person_emp_exp'],
        bins=emp_exp_bins,
        labels=emp_exp_labels
    )

    # Salva os bins para uso na inferência
    with open(os.path.join(output_dir, 'emp_exp_bins.pkl'), 'wb') as f:
        pickle.dump({'bins': emp_exp_bins, 'labels': emp_exp_labels}, f)

    ## Weighted Credit History Score
    new_df['weighted_credit_history'] = new_df['credit_score'] * new_df['cb_person_cred_hist_length']

    ## Income Category
    income_bins = pd.qcut(new_df['person_income'], q=4, retbins=True)[1]  # Retorna os limites
    income_labels = ['Low', 'Lower-Middle', 'Upper-Middle', 'High']
    new_df['income_category'] = pd.cut(
        new_df['person_income'],
        bins=income_bins,
        labels=income_labels,
        include_lowest=True
    )

    # Salva os bins para uso na inferência
    with open(os.path.join(output_dir, 'income_bins.pkl'), 'wb') as f:
        pickle.dump({'bins': income_bins, 'labels': income_labels}, f)

    ## Age Range
    new_df['age_range'] = new_df['person_age'].apply(lambda x: '20-30' if x < 30
                                                     else '30-40' if 30 <= x < 40
                                                     else '40-60' if 40 <= x < 60
                                                     else '> 60')

    ## Convertendo colunas categóricas
    for c in new_df.select_dtypes(include='object'):
        new_df[c] = new_df[c].astype('category')

    return new_df

import pandas as pd
import pickle
import os

def create_features_for_inference(data, metadata_dir="artifacts"):
    """
    Realiza a engenharia de features para dados de inferência usando os metadados salvos no treinamento.

    :param data: DataFrame com os dados de entrada.
    :param metadata_dir: Diretório onde os metadados (.pkl) foram salvos.
    :return: DataFrame com as novas features criadas.
    """
    # Carregar metadados salvos
    with open(os.path.join('..',metadata_dir, 'credit_score_bins.pkl'), 'rb') as f:
        credit_score_meta = pickle.load(f)
    
    with open(os.path.join('..',metadata_dir, 'emp_exp_bins.pkl'), 'rb') as f:
        emp_exp_meta = pickle.load(f)

    with open(os.path.join('..',metadata_dir, 'income_bins.pkl'), 'rb') as f:
        income_meta = pickle.load(f)

    # Criar features derivadas
    # Loan Income Ratio
    data['loan_income_ratio'] = data['loan_amnt'] / data['person_income']

    # Credit Score Category
    data['credit_score_category'] = pd.cut(
        data['credit_score'],
        bins=credit_score_meta['bins'],
        labels=credit_score_meta['labels']
    )

    # Employment Experience Groups
    data['emp_exp_group'] = pd.cut(
        data['person_emp_exp'],
        bins=emp_exp_meta['bins'],
        labels=emp_exp_meta['labels']
    )

    # Weighted Credit History Score
    data['weighted_credit_history'] = data['credit_score'] * data['cb_person_cred_hist_length']

    # Income Category
    data['income_category'] = pd.cut(
        data['person_income'],
        bins=income_meta['bins'],
        labels=income_meta['labels'],
        include_lowest=True
    )

    # Age Range
    data['age_range'] = data['person_age'].apply(lambda x: '20-30' if x < 30
                                                 else '30-40' if 30 <= x < 40
                                                 else '40-60' if 40 <= x < 60
                                                 else '> 60')

    # Garantir que as colunas categóricas sejam tratadas corretamente
    for c in data.select_dtypes(include='object').columns:
        data[c] = data[c].astype('category')

    return data
