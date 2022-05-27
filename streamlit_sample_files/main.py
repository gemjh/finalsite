import streamlit as st
from multiapp import MultiApp
import pandas as pd
import numpy as np
import plotly.express as px

def greeting1():
    st.title('정보 입력')

    st.selectbox('성별',('남','여'))
    st.text_input('나이')
    st.selectbox('학력',('초졸 이하','중졸','고졸 이상'))
    st.text_input('소득','0')
    st.text_input('아이 수','0')
    st.text_input('고용일수','0')
    st.selectbox('직장 전화',('있음','없음'))
    st.selectbox('휴대폰',('있음','없음'))
    st.selectbox('이메일',('있음','없음'))
    st.selectbox('자동차',('있음','없음'))
    st.selectbox('부동산',('있음','없음'))
    st.selectbox('소득 형태',('Commercial associate', 'Pensioner', 'State servant', 'Student', 'Working'))
    st.selectbox('가족 형태',('Civil marriage', 'Married', 'Separated', 'Single / not married', 'Widow'))
    st.selectbox('집 형태',('Co-op apartment',
    'House / apartment',
    'Municipal apartment',
    'Office apartment',
    'Rented apartment',
    'With parents'))
    st.selectbox('직업',('Accountants',
    'Cleaning staff',
    'Cooking staff',
    'Core staff',
    'Drivers',
    'HR staff',
    'High skill tech staff',
    'IT staff',
    'Laborers',
    'Low-skill Laborers',
    'Managers',
    'Medicine staff',
    'Private service staff',
    'Realty agents',
    'Sales staff',
    'Secretaries',
    'Security staff',
    'Waiters/barmen staff',
    '없음'))

    st.sidebar()

def greeting2():
    st.title('신용카드 사용자 신용도 예측 서비스')
    DATA_PATH = ('/Users/kij/projects/finalsite/data/final_df.csv')

    @st.cache
    def load_data(nrows):
        data = pd.read_csv(DATA_PATH, nrows=nrows)
        data.drop('Unnamed: 0', axis=1, inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(10)
    data_load_state.text("Done! (using st.cache)")

    st.subheader('그래프')

    #plotly bar차트
    data = data.groupby(by=['gender', 'credit_r']).count()
    data = data.groupby(level=0).apply(lambda x: x).reset_index()
    fig2 = px.bar(data, color_discrete_sequence=px.colors.qualitative.Pastel1,x='gender', y='credit_r', color='credit_r',
                    
                    category_orders={'gender': data['credit_r'].values,

                                    'credit_r':data['gender'].values},
                    labels={
                            'gender': 'gender',
                            "Unnamed: 0": "숫자",
                            'credit_r': 'credit_r'
                            },)
    st.plotly_chart(fig2)
    # st.plotly_chart(graph('credit_r','gender'))



    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)


app = MultiApp()
app.add_app("정보 입력", greeting1)
app.add_app("대시보드", greeting2)
app.run()