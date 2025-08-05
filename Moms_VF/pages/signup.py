import streamlit as st
hide_sidebar = """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="collapsedControl"] {
            display: none;
        }
        h3{
            text-align: center;
        }

        .stButton > button {
        height: 40px;
        border-radius: 5px;
        border: none;
        text-align: center;
        background-color: #4CAF50;
        color: white;
        font-family: Arial, sans-serif;
        font-size: 16px;
        transition: background-color 0.3s ease;
        margin-top: 10px;
        cursor: pointer;
        width: 100%;
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

st.markdown('<h3>Mom\'s VF Signup</h3>', unsafe_allow_html=True)
Name = st.text_input("Name", key="name_input")
username = st.text_input("Username", key="username_input")
password = st.text_input("Password", type="password", key="password_input")
email = st.text_input("Email", key="email_input")
if st.button("Sign Up"):
    if Name and username and password and email:
        st.success("Sign up successful!")
        st.markdown('<meta http-equiv="refresh" content="0; url=/login" />', unsafe_allow_html=True)
    else:
        st.error("Please fill all the required fields.")