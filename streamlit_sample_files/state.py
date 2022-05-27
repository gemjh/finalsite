import streamlit as st
from src.session import _get_state


state = _get_state()

if state.count == None:
    state.count = 0


def main():
    st.title('Counter App')

    increment_count = st.button('count +')
    decrement_count = st.button('count -')
    if increment_count:
        state.count += 1
    if decrement_count:
        state.count -= 1

    st.write(f'count: {state.count}')


if __name__ == '__main__':
    main()