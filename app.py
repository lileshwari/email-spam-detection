
import streamlit as st
import pickle

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Application title
st.title("Email Spam Detection")

st.write("Check whether your message is Spam or Not Spam.")

# Enter message
message = st.text_area("Enter your message:")

# Check button
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        result = model.predict([message])[0]

        if result == "spam":
            st.error("This is a SPAM message!")
        else:
            st.success("This is NOT SPAM!")
