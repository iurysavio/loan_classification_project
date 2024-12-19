import pandas as pd
import numpy as np

def create_features(df):
    df = df[df['person_emp_exp'] <= 76]
    # Implementing the new features:

    ## Loan Income Ratio
    df['loan_income_ratio'] = df['loan_amnt'] / df['person_income']

    ## Credit Score Category
    df['credit_score_category'] = pd.cut(
        df['credit_score']
        , bins= [300, 579, 669, 739, 799, 850]
        , labels= ['Very poor', 'Fair', 'Good', 'Very good', 'Exceptional']
    )

    ## Employment Experience Groups
    df['emp_exp_group'] = pd.cut(
        df['person_emp_exp']
        , bins= [-1, 2, 5, 10, float('inf')]
        , labels=['Entry-level', 'Mid-career', 'Experienced', 'Veteran']
    )

    ## Weighted Credit History Score
    df['weighted_credit_history'] = df['credit_score']*df['cb_person_cred_hist_length']

    ## Income category
    df['income_category'] = pd.qcut(
        df['person_income']
        , q= 4
        , labels=['Low', 'Lower-Middle', 'Upper-Middle', 'High']
    )

    ## Changing the types of categorical columns
    for c in df.select_dtypes(include='object'):
        df[c] = df[c].astype('category')
    
    return df