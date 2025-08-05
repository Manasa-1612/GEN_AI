import streamlit as st
# Page config
st.set_page_config(page_title="Login Page", page_icon=":lock:", layout="centered")
hide_sidebar = """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="collapsedControl"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# Title
st.markdown('<h3>Mom\'s VF Login</h3>', unsafe_allow_html=True)

username = st.text_input("Username", key="username_input")
password = st.text_input("Password", type="password", key="password_input")

if st.button("Login"):
    if username and password:
        st.success("Login successful!")
        st.markdown('<meta http-equiv="refresh" content="0; url=/user_dashboard" />', unsafe_allow_html=True)
    else:
        st.error("Please fill the required fields.")
# Signup link (✅ Corrected href)
else:
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <p>Don't you have an account?</p>
        <a href="/signup" target="_self">Sign Up</a>
    </div>
    """, unsafe_allow_html=True)

