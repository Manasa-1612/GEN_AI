import streamlit as st
import base64

# Hide sidebar using CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none;
        }
        h3 {
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Heading
st.markdown('<h3>Welcome to your Day</h3>', unsafe_allow_html=True)
# Encode your images
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Image paths
img1 = encode_image("D:/GEN_AI/langchain/Moms_VF/pages/health_remainder.jpeg")
img2 = encode_image("D:/GEN_AI/langchain/Moms_VF/pages/Meal_Planner.jpeg")
img3 = encode_image("D:/GEN_AI/langchain/Moms_VF/pages/house_management.png")
img4 = encode_image("D:/GEN_AI/langchain/Moms_VF/pages/story_tell.jpeg")

# Display images in 2 rows (3 images per row)
st.markdown(f"""
    <style>
            h3{{
            text-align: center;
            }}
        .image-grid {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 40px;
            margin-top: 40px;
        }}
        .image-grid a img {{
            width: 250px;
            height: 200px;
            border-radius: 20px;
            object-fit: cover;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .image-grid a img:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            cursor: pointer;
        }}
    </style>

    <div class="image-grid">
        <a href="/health_remainder" target="_new">
            <img src="data:image/png;base64,{img1}" alt="Health Reminder">
        </a>
        <a href="/Meal_Planner" target="_new">
            <img src="data:image/png;base64,{img2}" alt="Meal Planner">
        </a>
        <a href="/house_management" target="_new">
            <img src="data:image/png;base64,{img3}" alt="House Management">
        </a>
        <a href="/story_tells" target="_new">
            <img src="data:image/png;base64,{img4}" alt="Story Time">
        </a>
    </div>
""", unsafe_allow_html=True)

