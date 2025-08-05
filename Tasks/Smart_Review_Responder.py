import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from dotenv import load_dotenv


load_dotenv()

# Model
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(repo_id = "google/gemma-2-2b-it",
                          task = "text-generation")
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# Sentiment classification prompt
classifier_prompt = PromptTemplate(
    template=(
        "Classify the sentiment of the following text as 'positive', 'negative', or 'neutral'.\n\n"
        "Definitions:\n"
        "- Positive: Clearly expresses satisfaction or praise.\n"
        "- Negative: Clearly expresses dissatisfaction or criticism.\n"
        "- Neutral: Balanced, factual, or mentions both pros and cons.\n\n"
        "Only respond with the sentiment word.\n\n"
        "Text: {text}"
    ),
    input_variables=["text"]
)

classifier_chain = classifier_prompt | model | parser

# Feedback response prompts
positive_prompt = PromptTemplate(
    template="You are a customer support assistant. Based on the sentiment of the feedback, write a single appropriate response in a short, simple, and clear sentence. Do not suggest multiple responses or explain anything. Just respond once according to the positive sentiment Feedback:\n\n {text}",
    #template="Write an appropriate response to this positive feedback text:\n\n{text}",
    input_variables=["text"]
)

negative_prompt = PromptTemplate(
    template="You are a customer support assistant. Based on the negative sentiment of the feedback, write a single appropriate response in a short, simple, and clear sentence. Do not suggest multiple responses or explain anything. Just respond once according to the negative sentiment Feedback:\n\n {text}",
    #template="Write an appropriate response to this negative feedback text:\n\n{text}",
    input_variables=["text"]
)

neutral_prompt = PromptTemplate(
    template="You are a customer support assistant. Based on the sentiment of the feedback, write a single appropriate response in a short, simple, and clear sentence. Do not suggest multiple responses or explain anything. Just respond once according to the sentiment Feedback:\n\n {text}",
    #template="Write an appropriate response to this neutral feedback text:\n\n{text}",
    input_variables=["text"]
)

# Input text
#text_input = 'I like the design but the performance is poor.'
# Full sentiment-to-response chain
chain = (
    classifier_chain
    | RunnableLambda(lambda sentiment: {
        "sentiment": sentiment.strip().split()[0].lower(),  # Only use the first word
        "text": text
    })
    | RunnableBranch(
        (lambda x: x["sentiment"] == "positive", positive_prompt | model | parser),
        (lambda x: x["sentiment"] == "negative", negative_prompt | model | parser),
        (lambda x: x["sentiment"] == "neutral",  neutral_prompt | model | parser),
        (lambda x: f"Sentiment not recognized: {x['sentiment']}")
    )
)
#output = chain.invoke(text_input)
#print("Generated Response:\n", output)
st.title("🧠 Smart Review Responder")
text = st.text_input("📝 Enter your feedback text:")
if st.button("Submit Feedback"):
    if text:
        with st.spinner(" 📊 Generating response..."):
            try:
                response = chain.invoke({"text": text})
                if response:
                    st.success("✅ Response generated successfully!")
                    st.write("Response:", response)
                else:
                    st.warning("No response generated.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Fill all the required Fields")