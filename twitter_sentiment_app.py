import streamlit as st
from model_inference import predict_sentiment  # remove .py

# Set up Streamlit app
st.title("Sentiment Analysis App")

# Text input for user
user_input = st.text_area("Enter your text: (Less than 30 words)")

if st.button("Predict"):
    if user_input:
        # Call the prediction function
        prediction = predict_sentiment(user_input)

        # Display the result
        st.write(f"Sentiment: {prediction.title()}")
    else:
        st.warning("Please enter some text for analysis.")
