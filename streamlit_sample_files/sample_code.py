import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from persist import persist, load_widget_state

st.title('신용카드 사용자 신용도 예측 서비스')

with st.sidebar:
    st.subheader('정보를 입력해주세요.')

    gender = st.selectbox('성별을 선택해주세요. (F : 여자, M : 남자)', ('F', 'M'))
    st.write('성별:', gender)

    car = st.selectbox('차량 소유 여부를 선택해주세요. (Y : 소유, M : 미소유)', ('Y', 'N'))
    st.write('차량 소유 여부:', car)

    reality = st.selectbox('부동산 소유 여부를 선택해주세요. (Y : 소유, M : 미소유)', ('Y', 'N'))
    st.write('부동산 소유 여부:', reality)

    child_num = st.selectbox('자녀 수를 선택해주세요. (5명 이상 시 5를 선택해주세요.)', (1, 2, 3, 4, 5))
    st.write('자녀 수:', child_num)

    income_total = st.selectbox('연간 소득을 선택해주세요.', (100, 1000))
    st.write('연간 소득:', income_total)

    income_type = st.selectbox('소득 분류를 선택해주세요.', ('Commercial associate', 'Working', 'State servant', 'Pensioner', 'Student'))
    st.write('소득 분류:', income_type)

    edu_type = st.selectbox('학력을 선택해주세요.', ('Higher education' ,'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree'))
    st.write('학력:', edu_type)

    family_type = st.selectbox('결혼 여부를 선택해주세요.', ('Married', 'Civil marriage', 'Separated', 'Single / not married', 'Widow'))
    st.write('결혼 여부:', family_type)

    house_type = st.selectbox('생활 방식을 선택해주세요.', ('Municipal apartment', 'House / apartment', 'With parents', 'Co-op apartment', 'Rented apartment', 'Office apartment'))
    st.write('생활 방식:', family_type)

    DAYS_BIRTH = st.text_input('나이를 입력해주세요.', '')
    st.write('나이:', DAYS_BIRTH)

    # st.text_input('')


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
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# hist_values = np.histogram(data['credit_r'], data['occyp_type'].value_counts().index)
# st.bar_chart(hist_values)

# st.plotly_chart(data, use_container_width=False, sharing="streamlit")
# import plotly.graph_objects as go
# fig = go.Figure(data)
# fig.show()
# st.plotly_chart(fig, use_container_width=True)

def graph(col1,col2):
    df = data.groupby(by=[col1,col2]).count()
    # df = df.groupby(level=0).apply(lambda x: 100 * x / x.sum()).reset_index()
    df = df.groupby(level=0).apply(lambda x: x).reset_index()

    # fig = px.bar(df, color_discrete_sequence=["rgb(179,226,205)", "rgb(203,213,232)"],x=col1, y='Unnamed: 0', color=col2,
    fig = px.bar(df, color_discrete_sequence=px.colors.qualitative.Pastel1,x=col1, y='Unnamed: 0', color=col2,
                
                category_orders={col1: data[col1].values,

                                col2:data[col2].values},
                labels={
                        col1: col1,
                        "Unnamed: 0": "숫자",
                        col2: col2
                        },)

    fig.update_layout(title='그래프')
    fig.show()

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