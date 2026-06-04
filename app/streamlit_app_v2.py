import streamlit as st
import pickle
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.markdown("""
<style>
.stApp {
    background-color: #EAF4FF;
}

h1 {
    color: #1565C0;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)
# --- FILE PATH SETUP ---
# Using Pathlib ensures the app can find the files regardless of the environment
BASE_DIR = Path(__file__).parent

# --- LOAD MODEL AND VECTORIZER ---
@st.cache_resource
def load_assets():
    try:
        model_path = BASE_DIR / "model.pkl"
        vectorizer_path = BASE_DIR / "vectorizer.pkl"
        
        with open(model_path, 'rb') as m_file:
            model = pickle.load(m_file)
        with open(vectorizer_path, 'rb') as v_file:
            vectorizer = pickle.load(v_file)
            
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_assets()

# --- USER INTERFACE ---
st.title("📧 Spam Message Classifier")
st.write("Enter a message below to check if it's Spam or Not Spam.")

if model is None or vectorizer is None:
    st.error(f"Model files not found in: {BASE_DIR}")
    st.info("Ensure 'model.pkl' and 'vectorizer.pkl' are in the same folder as this script.")
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
st.caption("Built with Streamlit, Scikit-Learn, and Pathlib")
