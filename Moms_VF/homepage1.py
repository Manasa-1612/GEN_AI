import streamlit as st
import base64

# Hide sidebar using CSS
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

# Encode and display circular image
image_path = "D:/GEN_AI/langchain/Moms_VF/generated-image.png"
with open(image_path, "rb") as img_file:
    encoded = base64.b64encode(img_file.read()).decode()

# Custom CSS
st.markdown("""
    <style>
    .circular-img {
        margin-top: 50px;
        width: 400px;
        height: 400px;
        border-radius: 30%;
        object-fit: cover;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .circular-img:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        cursor: pointer;
        background-color: black;
    }
    .center {
        text-align: center;
        #background-color: lightgreen;
    }
    *{
            padding :0;
            margin: 0;
            box-sizing: border-box;
            background-color: lightgrey;}
   img{
            background-color: white;}
    </style>
""", unsafe_allow_html=True)

# Centered title and image
#st.markdown("<h1 class='center'>Mom's VF</h1>", unsafe_allow_html=True)

# Clickable image that navigates to /login
st.markdown(f"""
    <div class="center">
        <a href="/login2" target="_self">
            <img src="data:image/jpeg;base64,{encoded}" class="circular-img" alt="Mom's VF Logo">
        </a>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h6 class='center'>Click the Image to Login</h6>", unsafe_allow_html=True)
