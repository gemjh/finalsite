import streamlit as st
# from multiapp import MultiApp
import pandas as pd
import numpy as np
import plotly.express as px
from persist import persist, load_widget_state
# from catboost_model_sample import preprocessing, model


# 메뉴 번역
# 메뉴 선택 방식 (라디오버튼, 셀렉트박스, 텍스트)


def main():
    if "page" not in st.session_state:
        # Initialize session state.
        st.session_state.update({
            # Default page.
            "page": "home",

            # Radio, selectbox and multiselect options.
            "gender_options": ["F", "M"],

            # 기본값
            "text": "",
            "slider": 0,
            "checkbox": False,
            "radio": "F",
            "selectbox": "Hello",
            "multiselect": ["Hello", "Everyone"],
        })

    page = st.sidebar.radio("Select your page", tuple(PAGES.keys()), format_func=str.capitalize)

    PAGES[page]()


def total_graph():
    st.subheader('전체 표 보기')
    # st.title('신용카드 사용자 신용도 예측 서비스')
    DATA_PATH = ('/Users/kij/projects/finalsite/data/')

    @st.cache
    def load_data(nrows):
        train = pd.read_csv(DATA_PATH + 'final_df.csv', nrows=nrows)
        test = pd.read_csv(DATA_PATH + 'final_test_df.csv', nrows=nrows)
        train.drop('Unnamed: 0', axis=1, inplace=True)
        return train, test

    data_load_state = st.text('Loading data...')
    train, test = load_data(10)
    data_load_state.text("Done! (using st.cache)")


    #plotly bar차트
    # data = train.groupby(by=['gender', 'credit_r']).count()
    # data = train.groupby(level=0).apply(lambda x: x).reset_index()
    # fig2 = px.bar(data, color_discrete_sequence=px.colors.qualitative.Pastel1,x='gender', y='credit_r', color='credit_r',
                    
    #                 category_orders={'gender': data['credit_r'].values,

    #                                 'credit_r':data['gender'].values},
    #                 labels={
    #                         'gender': 'gender',
    #                         "Unnamed: 0": "숫자",
    #                         'credit_r': 'credit_r'
    #                         },)
    # st.plotly_chart(fig2)



    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(train)

        
        # st.write(preprocessing(train, test))
        # st.write(train_model(data))
        # st.write(result())


    st.write(
        f"""
        Settings values
        ---------------
        - **Gender**: {st.session_state.gender}
        - **Slider**: `{st.session_state.slider}`
        - **Checkbox**: `{st.session_state.checkbox}`
        - **Radio**: {st.session_state.radio}
        - **Selectbox**: {st.session_state.selectbox}
        - **Multiselect**: {", ".join(st.session_state.multiselect)}
        """
    )



def my_settings():
    st.header("정보를 입력해주세요")

    st.radio('성별', st.session_state["gender_options"], key=persist("gender"))
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


def my_graph():
    st.header("내 표 보기")


PAGES = {
    "내 정보 입력": my_settings,
    "전체 표 보기": total_graph,
    "내 표 보기" : my_graph
}


if __name__ == "__main__":
    load_widget_state()
    main()
    # preprocessing()
    # train_model()
    # result()