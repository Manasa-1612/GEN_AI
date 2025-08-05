import streamlit as st
import groq
st.set_page_config(page_title="Meal Planner", layout="centered")
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
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("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")

client = Groq(api_key=GROQ_API_KEY)


st.title("🥗 Mom's Virtual Friend - Meal Planner")

# --- Input Fields ---
ingredients = st.text_area("📝 List the ingredients you have:", placeholder="e.g., eggs, tomatoes, spinach, rice")

dietary = st.multiselect(
    "💡 Dietary Preferences:",
    ["Vegetarian", "Vegan", "Gluten-Free", "Keto", "Halal", "Dairy-Free"]
)

meal_type = st.selectbox("🍽️ Select Meal Type:", ["Breakfast", "Lunch", "Dinner", "Snack"])

time_available = st.slider("⏳ Time Available to Cook (in minutes):", 5, 120, 30)

servings = st.number_input("👨‍👩‍👧‍👦 Servings Required:", min_value=1, step=1, value=2)

cuisine = st.selectbox(
    "🌍 Cuisine Preference (Optional):",
    ["Any", "Andhrapradesh", "Gujarat","Karnataka", "Kerala", "Maharashtra", "Telangana", "Tamil Nadu"]
)

# --- Prompt Construction ---
def build_prompt():
    base = f"""
You are a smart meal planning assistant. Given the ingredients: {ingredients},
generate 3 simple and quick recipes for {meal_type.lower()} that can be prepared within {time_available} minutes 
for {servings} servings.

Consider these preferences:
- Dietary: {', '.join(dietary) if dietary else 'None'}
- Cuisine: {cuisine}

For each recipe, include:
1. Recipe Name
2. Ingredients (mention if anything is missing)
3. Short Instructions
4. Estimated Prep & Cook Time
5. Suggest 2 extra ingredients that would improve the dish

Make the response clear, user-friendly, and concise.
"""
    return base.strip()

# --- GenAI Integration ---
def get_recipes_from_groq(prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=1024,
    )
    return response.choices[0].message.content

# --- Output ---
if st.button("🍳 Generate Recipes"):
    if not ingredients.strip():
        st.warning("Please enter at least some ingredients.")
    else:
        with st.spinner("Thinking..."):
            prompt = build_prompt()
            response = get_recipes_from_groq(prompt)
            st.markdown("### 📋 Recommended Recipes")
            st.markdown(response)