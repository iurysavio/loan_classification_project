import streamlit as st
import json
import requests


API_URL = 'http://127.0.0.1:8000/predict'
st.set_page_config(layout="wide")

mapping_dict_pt = {
    'person_gender' : {
        'Masculino': 'male',
        'Feminino': 'female' 
    },
    'person_education': {
        'Mestrado' : 'Master',
        'Ensino Médio' : 'High School',
        'Ensino Superior' : 'Bachelor',
        'Tecnólogo': 'Associate',
        'Doutorado': 'Doctorate'
    },
    'person_home_ownership': {
        'Aluguel':'RENT',
        'Moradia própria': 'OWN',
        'Hipoteca': 'MORTGAGE',
        'Outro': 'OTHER'
    },
    'loan_intent': {
        'Pessoal': 'PERSONAL',
        'Educação': 'EDUCATION',
        'Necessidades médicas': 'MEDICAL',
        'Empreendimentos':'VENTURE',
        'Reforma/ Construção': 'HOMEIMPROVEMENT',
        'Consolidação de Dívidas':'DEBTCONSOLIDATION'
    },
    'previous_loan_defaults_on_file' : {
        'Sim' : 'Yes',
        'Não' : 'No'
    }
}

st.sidebar.title('Teste')
st.markdown(
    """
# Loan Approval Machine Learning Inference Project
"""
)

# Criação de um Radio Button com opções horizontais
selected_option = st.radio(
    "Escolha uma opção:",
    options=["PT-BR", "EN-US"],
    horizontal=True  # Define o layout como horizontal
)

if selected_option == 'PT-BR':
    # Exibindo a opção selecionada
    st.write(f"O projeto em questão traz de forma sucinta uma aplicação end-to-end que propõe a utilização de algumas ferramentas e conceitos do mundo de dados.")
    
    col1, col2, col3= st.columns([1, 1, 1])  # Ajuste os valores para controlar o espaço
    with col1:
        previous_loan_defaults_on_file_pt= st.selectbox(
        'O cliente tem histórico de inadimplência?'
        , ['Sim', 'Não'])
        previous_loan_defaults_on_file = mapping_dict_pt['previous_loan_defaults_on_file'][previous_loan_defaults_on_file_pt]

        # PERSON AGE
        person_age = st.slider("Idade do cliente?", 0, 90, 37,key="unique_slider_key")
        
        # PERSON GENDER
        person_gender_pt = st.selectbox(
            'Qual gênero do cliente?',
            ['Masculino', 'Feminino']
        )
        person_gender = mapping_dict_pt['person_gender'][person_gender_pt]
        
        
        # PERSON EDUCATION
        person_education_pt = st.selectbox(
             "Qual o nível educacional do cliente?"
            , ['Ensino Médio', 'Tecnólogo', 'Ensino Superior', 'Mestrado', 'Doutorado']
        )
        person_education = mapping_dict_pt['person_education'][person_education_pt]
        

        # PERSON EMPLOYER EXPERIENCE
        person_emp_exp = st.number_input(
            'Quantos anos de experiência de trabalho o cliente tem?',min_value=0, max_value= 50
        )
        

        # PERSON HOME OWNERSHIP
        person_home_ownership_pt = st.selectbox(
            'Qual o seu tipo de moradia?',
            ['Aluguel','Moradia própria','Hipoteca','Outro']
        )
        person_home_ownership = mapping_dict_pt['person_home_ownership'][person_home_ownership_pt]
        
    with col2:
        # LOAN AMOUNT
        loan_amnt = st.number_input(
            'Quanto você necessita de empréstimo?', min_value=0, max_value=100000, value=20000, step= 1
        )
        

        # PERSON INCOME
        person_income = 12 * st.number_input(
            'Qual a sua renda mensal?', min_value=0, max_value=100000, value=10000, step= 1
        )
        

        # LOAN INTENT
        loan_intent_pt = st.selectbox(
            'Qual o intuito do empréstimo?',
            ['Pessoal','Educação','Necessidades médicas','Empreendimentos','Reforma/ Construção','Consolidação de Dívidas']
        )
        loan_intent = mapping_dict_pt['loan_intent'][loan_intent_pt]
        
        # LOAN PERCENT INCOME
        loan_percent_income = round(loan_amnt/person_income,2)

        
        # LOAN ANNUAL TAX
        loan_int_rate = st.number_input(
            'Qual é a taxa de juros anual do empréstimo?'
            , min_value= 0.00
            , max_value= 20.00
            , step = 0.01
        )

        # CREDIT HISTORY TIME LENGHT
        cb_person_cred_hist_length = st.number_input(
            'Qual tempo de crédito que o cliente possui? (em anos)'
            , min_value = 0,max_value= 5, value= 2, step=1
        )

        # CREDIT SCORE
        credit_score = st.number_input(
            'Qual o score de crédito do cliente?'
            , min_value = 300,max_value= 850, value= 500, step=50
        )


def send_info_to_API(
          person_age
        , person_gender
        , person_education
        , person_income
        , person_emp_exp
        , person_home_ownership
        , loan_amnt
        , loan_intent
        , loan_int_rate
        , loan_percent_income
        , cb_person_cred_hist_length
        , credit_score
        , previous_loan_defaults_on_file) -> requests.Response:
    """
    Sends image to the API endpoint for prediction.
    """
    body_json = {
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }
    response = requests.post(API_URL, json=body_json) 
    response_dict = json.loads(response.text)
    return response_dict['prediction']

predict_button = st.button('Predict')
    
if predict_button:  
    prediction = send_info_to_API(person_age
        , person_gender
        , person_education
        , person_income
        , person_emp_exp
        , person_home_ownership
        , loan_amnt
        , loan_intent
        , loan_int_rate
        , loan_percent_income
        , cb_person_cred_hist_length
        , credit_score
        , previous_loan_defaults_on_file)
 
    if prediction == 1:
        st.markdown(f'<div class= "form-box" <p class="button-font"> O empréstimo para o cliente foi APROVADO! </div>', unsafe_allow_html=True)
    if prediction == 0:
        st.markdown(f'<div class= "form-box" <p class="button-font"> O empréstimo para o cliente foi NEGADO! </div>', unsafe_allow_html=True)
        