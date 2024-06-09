import streamlit as st
from langchain_helper import few_shot_db_chain

st.title("4Tees Tshirt: Database QnA")
question = st.text_input("Question: ")
if question:
    chain = few_shot_db_chain()
    answer = chain.run(question)
    st.header("Answer: ")
    st.write(answer)


