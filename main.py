import streamlit as st
from langgraph_component import graph_output


st.title("Essay Writer")

question = st.text_input("Give topic for essay")
btn = st.button("write essay")

if btn:
    if question:
        text = graph_output(question)
        st.write(text)
    else:
        st.header("Please give topic for essay")