import streamlit as st
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

# --- LOAD MODEL AND VECTORIZER ---
# We use @st.cache_resource to load the model only once, speeding up the app
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_assets()

# --- USER INTERFACE ---
st.title("📧 Spam Message Classifier")
st.write("Enter a message below to check if it's Spam or Not Spam.")

if model is None or vectorizer is None:
    st.error("Model files ('model.pkl', 'vectorizer.pkl') not found in the directory.")
    st.info("Please upload your model and vectorizer files to the same repository/folder.")
else:
    # Text input area
    text_input = st.text_area("Message content:", placeholder="Type your message here...", height=150)

    if st.button("Predict"):
        if text_input.strip() == "":
            st.warning("Please enter some text before predicting.")
        else:
            # Transformation and Prediction
            vector_input = vectorizer.transform([text_input])
            result = model.predict(vector_input)[0]

            # Display Results
            if result == 1:
                st.error("### Prediction: **Spam**")
            else:
                st.success("### Prediction: **Not Spam**")

# Optional footer
st.divider()
st.caption("Built with Streamlit and Scikit-Learn")
