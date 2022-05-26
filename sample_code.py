import streamlit as st
import pandas as pd
import numpy as np

st.title('신용카드 사용자 신용도 예측 서비스')

# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)


with st.sidebar:
    st.subheader('정보를 입력해주세요.')

    gender = st.selectbox('성별을 선택해주세요. (F : 여자, M : 남자)', ('F', 'M'))
    st.write('성별:', gender)

    car = st.selectbox('차량 소유 여부를 선택해주세요. (Y : 소유, M : 미소유)', ('Y', 'N'))
    st.write('차량 소유 여부:', car)

    reality = st.selectbox('부동산 소유 여부를 선택해주세요. (Y : 소유, M : 미소유)', ('Y', 'N'))
    st.write('부동산 소유 여부:', reality)

    child_num = st.selectbox('자녀 수를 선택해주세요. (7명 이상 시 7을 선택해주세요.)', (1, 2, 3, 4, 5, 6, 7))
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

    DAYS_BIRTH = st.text_input('')
    st.write('나이:', DAYS_BIRTH)

    # st.text_input('')


# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#             'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
DATA_PATH = ('/Users/kij/projects/finalsite/data/final_df.csv')

@st.cache
def load_data(nrows):
    # data = pd.read_csv(DATA_URL, nrows=nrows)
    data = pd.read_csv(DATA_PATH, nrows=nrows)
    data.drop('Unnamed: 0', axis=1, inplace=True)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
# data = load_data(10000)
data = load_data(10)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


st.subheader('그래프')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
hist_values = np.histogram(data['credit_r'], data['occyp_type'].value_counts().index)
st.bar_chart(hist_values)
