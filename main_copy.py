import streamlit as st
# from multiapp import MultiApp
import pandas as pd
import numpy as np
import plotly.express as px
from persist import persist, load_widget_state
from catboost_model_sample import preprocessing, train_model, result
from catboost import CatBoostClassifier # 05/28


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
    # DATA_PATH = ('/Users/kij/projects/finalsite/data/')
    # DATA_PATH = ('/Users/jiho/Desktop/mulcam_final/data/')
    DATA_PATH = ('./data/')

    # # train 차트 불러오기
    # @st.cache
    # def load_train(nrows):
    #     train = pd.read_csv(DATA_PATH + 'final_df.csv', nrows=nrows)
    #     # train = pd.read_csv(DATA_PATH + 'final_df.csv')
    #     train.drop(['credit_r', 'DAYS_EMPLOYED'], axis=1, inplace=True)
    #     train.drop('Unnamed: 0', axis=1, inplace=True)
    #     return train

    # # test 차트 불러오기
    # @st.cache
    # def load_test(nrows):
    #     test = pd.read_csv(DATA_PATH + 'final_test_df.csv', nrows=nrows)
    #     # test = pd.read_csv(DATA_PATH + 'final_test_df.csv')
    #     test.drop(['DAYS_EMPLOYED'], axis=1, inplace=True)
    #     return test

    # 불러온 차트 보여주기
    # data_load_state = st.text('Loading data...')
    train = pd.read_csv(DATA_PATH + 'final_df.csv')
    train.drop(['credit_r', 'DAYS_EMPLOYED'], axis=1, inplace=True)
    test = pd.read_csv(DATA_PATH + 'final_test_df.csv')
    test.drop(['DAYS_EMPLOYED'], axis=1, inplace=True)
    # test = load_test(10)
    # data_load_state.text("Done! (using st.cache)")


# income_total, income_type, days_birth, days_employed,
# occyp_type, begin_month, ability, income_mean 


    #plotly bar차트
    # train_data = train.groupby(by=['gender', 'credit']).count()
    # train_data = train.groupby(level=0).apply(lambda x: x).reset_index()

    # fig2 = px.bar(train, color_discrete_sequence=px.colors.qualitative.Pastel1,x='gender', y='credit', color='credit',
                    
    #                 category_orders={'gender': train_data['credit'].values,

    #                                 'credit':train_data['gender'].values},
    #                 labels={
    #                         'gender': 'gender',
    #                         "Unnamed: 0": "숫자",
    #                         'credit': 'credit'
    #                         },)
    # st.plotly_chart(fig2)

    #income_total
    for col1, col2 in [['income_total', 'DAYS_BIRTH']]:
        df = train.copy()
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col2, y=col1, color="credit", title=str('Scatter chart: '+col1+' & '+col2))
                        #  size='', hover_data=[''])
        # fig.show()
        st.plotly_chart(fig)
    
    #income_type
    for col in ['income_type']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.bar(df, x=col, y="Unnamed: 0", title=str('Bar chart: '+col+' & credit count'), color="credit", labels={col:'income_type','Unnamed: 0':'credit count'})
        st.plotly_chart(fig)

    #days_birth
    for col in ['DAYS_BIRTH']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.bar(df, x=col, y="Unnamed: 0", title=str('Bar chart: '+col+' & credit count'), color="credit", labels={col:'age','Unnamed: 0':'credit count'})
        st.plotly_chart(fig)

    #DAYS_EMPLOYED
    #train['DAYS_EMPLOYED_Y'] = train['DAYS_EMPLOYED_Y'].map(lambda x: 0 if x < 0 else x)
    for col in ['DAYS_EMPLOYED_r']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col, y="Unnamed: 0", title=str('Bar chart: '+col+' & credit count'), color="credit", labels={col:'DAYS_EMPLOYED_r','Unnamed: 0':'credit count'})
        st.plotly_chart(fig)

    #occyp_type
    for col in ['occyp_type']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.pie(df, values='Unnamed: 0', names=col, title=str('Pie chart: '+col), labels={col:'occyp_type','Unnamed: 0':'credit count'})
        # fig.show()
        st.plotly_chart(fig)

    #begin_month
    for col in ['begin_month']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.bar(df, x=col, y="Unnamed: 0", title=str('Bar chart: '+col+' & credit count'), color="credit", labels={col:'begin_month','Unnamed: 0':'credit count'})
        # fig.show()
        st.plotly_chart(fig)
    
    # 데이터 전처리
    train.drop('Unnamed: 0', axis=1, inplace=True)
    pre_train, pre_test = preprocessing(train, test)


    #ability
    for col1, col2 in [['ability', 'income_type']]:
        df = train.copy()
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col2, y=col1, color="credit", title=str('Scatter chart: '+col1+' & '+col2))
                        #  size='', hover_data=[''])
        # fig.show()
        st.plotly_chart(fig)
    
    #income_mean
    for col1, col2 in [['income_mean', 'family_type']]:
        df = train.copy()
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col2, y=col1, color="credit", title=str('Scatter chart: '+col1+' & '+col2))
                        #  size='', hover_data=[''])
        # fig.show()
        st.plotly_chart(fig)
    
    # sunburst chart
    # col1, col2, col3 in 범주형 변수, 범주형 변수, 범주형 변수
    for col1, col2, col3 in [['family_type','income_type', 'credit']]:
        df = train.copy()
        fig = px.sunburst(df, path=[col1,col2,col3], title = str('Sun chart: '+col1+' & '+col2+' & '+col3))
        # fig.show()
        st.plotly_chart(fig)

    # if st.checkbox('Show raw data'):
    #     st.subheader('Raw data')
    #     st.write(train)
    #     st.write(test)

    train = pd.read_csv(DATA_PATH + 'final_df.csv')
    # train.drop('Unnamed: 0', axis=1, inplace=True)
    X_train = pd.read_csv(DATA_PATH + 'service.csv')
    # pre_train, pre_test = preprocessing(train, test)
    # pre_train, pre_X_train = preprocessing(train, X_train)
    preprocessing(train, X_train)
    # 데이터 전처리 출력
    # st.write(pre_train, pre_test)
    # st.write(pre_train, pre_X_train)


    # 모델 학습
    # model_cat, X_train = train_model(pre_train, pre_test)
    # model_cat.save_model('/Users/jiho/Desktop/mulcam_final/data/model.bin') # 5/28
    # model_cat.save_model("./data/model.bin")

    # 모델 학습 출력
    # st.write(model_cat, X_train)
    from_file = CatBoostClassifier()  # 5/28
    # from_file.load_model("/Users/jiho/Desktop/mulcam_final/data/model.bin") # 5/28
    from_file.load_model("./data/model.bin")

    # 학습 결과
    # y_predict = result(model_cat, X_train)
    y_predict = from_file.predict(X_train) # 5/28
    # 학습 결과 출력
    st.write(y_predict)


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
    st.selectbox('학력',('Higher education' ,'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree'))
    st.text_input('소득','0')
    st.text_input('아이 수','0')
    st.text_input('고용연수','0')
    st.radio('직장 전화',('있음','없음'))
    st.radio('집 전화',('있음','없음'))
    st.radio('이메일',('있음','없음'))
    st.radio('자동차',('있음','없음'))
    st.radio('부동산',('있음','없음'))
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
        'jobless'))
    st.text_input('가족 규모','0')
    # st.text_input('신용카드 발급 개월수', '0')


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