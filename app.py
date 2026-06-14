import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer

@st.cache_resource
def load_models():
    model = load_model(r'C:\Users\ASUS\OneDrive\Desktop\Deep Learning BASIC TO ADV\Projects\sentiment-analyzer\sentiment-analyzer.keras')
    with open(r'C:\Users\ASUS\OneDrive\Desktop\Deep Learning BASIC TO ADV\Projects\sentiment-analyzer\tokenizer.pkl', 'rb') as f: tok = pickle.load(f)
    
    return model, tok

model, tok = load_models()
# model = load_model(r'C:\Users\ASUS\OneDrive\Desktop\Deep Learning BASIC TO ADV\Projects\sentiment-analyzer\sentiment-analyzer.keras')
# with open(r'C:\Users\ASUS\OneDrive\Desktop\Deep Learning BASIC TO ADV\Projects\sentiment-analyzer\tokenizer.pkl', 'rb') as f:
#     tok = pickle.load(f)

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭")
st.title("🎭 Sentiment Analyzer")
st.write("Enter any tweet or text to analyze its sentiment")

user_input = st.text_area("Your text here:", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        seq = tok.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=120)
        pred = model.predict(padded)[0][0]
        
        if pred > 0.5:
            st.success(f"😊 Positive Sentiment ({pred*100:.1f}% confidence)")
        else:
            st.error(f"😔 Negative Sentiment ({(1-pred)*100:.1f}% confidence)")