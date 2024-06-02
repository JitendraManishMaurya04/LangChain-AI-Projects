import streamlit as st
import langchain_helper as lh

st.title("Restraunt Name Generator")

cuisine = st.sidebar.selectbox("Pick a Cuisine", ("Indian", "Italian", "Mexican", "American", "Chinese"))

if cuisine:
    response = lh.generate_restrauntName_and_menuItems(cuisine)
    st.header(response['restraunt_name'].strip())

    st.write("***MENU ITEMS***")
    menu_items = response['menu_items'].strip().split(",")
    for item in menu_items:
        st.write("-->",item)

