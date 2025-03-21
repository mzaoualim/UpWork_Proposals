
import streamlit as st
import shap
from streamlit_shap import st_shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib

# loading the model
try:
    model = joblib.load('Machine_learning/loan_approval/app/model_lgbm.pkl')
except Exception as e:
    st.error(f'An error occured: {e}')
    # st.stop()

def main():
    st.markdown("<h2 style='text-align: center;'> INPUTS </h2>", unsafe_allow_html=True)

    with st.form(key='loan_form'):
        a1, a2, a3 = st.columns(3)
        with a1:
            person_age = st.number_input('Age', min_value=0, max_value=100, value=20)
        with a2:
            person_income = st.number_input('Income', min_value=0, value=50000)
        with a3:
            person_emp_length = st.number_input('Employement length', min_value=0, value=5)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            loan_percent_income = st.number_input('Loan to Income Share', min_value=0.00, max_value=1.00, value=0.10)
            cb_person_default_on_file = st.selectbox('Previous Default', ['Y', 'N'])

        with c2:
            loan_amnt = st.number_input('Loan Amount', min_value=0, value=10000)
            loan_grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])

        with c3:
            cb_person_cred_hist_length = st.number_input('Credit History Length', min_value=0, value=7)
            person_home_ownership = st.selectbox('Home Ownership', ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])

        with c4:
            loan_int_rate = st.number_input('Loan Interest Rate', min_value=0.00, value=5.00)
            loan_intent = st.selectbox('Loan Intent', ['EDUCATION', 'MEDICAL', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])

        submit_button = st.form_submit_button("Predict", use_container_width=True)

        if submit_button:
            input_data = {
                'person_age': person_age,
                'person_income': person_income,
                'person_home_ownership': person_home_ownership,
                'person_emp_length': person_emp_length,
                'loan_intent': loan_intent,
                'loan_grade': loan_grade,
                'loan_amnt': loan_amnt,
                'loan_int_rate': loan_int_rate,
                'loan_percent_income': loan_percent_income,
                'cb_person_default_on_file': cb_person_default_on_file,
                'cb_person_cred_hist_length': cb_person_cred_hist_length,
            }

            # inputs preprocessing
            input_df = pd.DataFrame([input_data])
            input_df[input_df.select_dtypes(include='object').columns] = input_df[input_df.select_dtypes(include='object').columns].astype('category')

            # Predict
            prediction = model.predict(input_df)
            result = 'Approved' if prediction[0] == 1 else 'Denied'
            st.write('---')
            
            st.markdown("<h2 style='text-align: center;'> OUTPUT </h2>", unsafe_allow_html=True)

            st.success('Your requested loan is {}'.format(result))
            st.write('---')

            st.markdown("<h2 style='text-align: center;'> SHAP </h2>", unsafe_allow_html=True)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_df)

            force_fig = shap.force_plot(shap_values.base_values[0], 
                                        shap_values.values[0],
                                        input_df.columns,
                                        matplotlib=True, 
                                        show=False)

            st.pyplot(force_fig, use_container_width=True)

            st.write('---')

            st.markdown("<h2 style='text-align: center;'> LLM COMMENT </h2>", unsafe_allow_html=True)
            st.write(input_data)
            st.write(shap_values)
            st.write('---')











if __name__ == '__main__':
   main()