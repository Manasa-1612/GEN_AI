# Task-4 Mini Personal Task Assistant App - LangChain + Ollama + Streamlit

# This code creates a simple Streamlit app that uses LangChain with Ollama to answer user questions.

# Step-1: Import necessary libraries
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


#Step-2: Prepare the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
       ]
)

# Step-3: Set up the Streamlit app
st.header("AI Personal Task Assistant App")

# Step-4: Create a text input for user's name and user's questions
user_name = st.text_input("Enter your name:")
user_input = st.text_input("Enter your question:")

# Step-5: Initialize the Ollama LLM and output parser
llm = OllamaLLM(model="llama3.2:1b")
parser = StrOutputParser()

# Step-6: Create a chain that combines the prompt, LLM, and parser
chain = prompt | llm | parser

# Step-7: If user input is provided, invoke the chain and display the response
if st.button("Generate Response"):
    if user_name and user_input:
        with st.spinner("Generating response..."):
            try:
                response = chain.invoke({"input": f"{user_name}. {user_input}"})
                #response = chain.invoke({"input":  user_input})
                if response:
                    st.write("Response:", response)
                else:
                    st.warning("No response generated.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Fill all the required Fields")