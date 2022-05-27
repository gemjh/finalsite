import streamlit as st
from multiapp import MultiApp

def greeting1():
    st.title('こんにちは, 世界！')
    st.write('ねこはかわいい')

def greeting2():
    st.title('またまたこんにちは, 世界！！')
    st.write('ねこはとてもかわいい')


app = MultiApp()
app.add_app("page1", greeting1)
app.add_app("page2", greeting2)
app.run()