import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_PATH = "models/bert_finetuned"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

LABELS = {0: "Negative", 1: "Positive"}

st.set_page_config(page_title="Türkçe Duygu Analizi", page_icon="💬")
st.title("💬 Türkçe Duygu Analizi")
st.write("Bir cümle yaz, fine-tune edilmiş Türkçe BERT modeli duygusunu tahmin etsin.")

text = st.text_area("Cümlenizi girin:", placeholder="Örnek: Bu ürün gerçekten harika, kesinlikle tavsiye ederim!")

if st.button("Tahmin Et") and text.strip():
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pred_class = torch.argmax(probs).item()
    pred_label = LABELS[pred_class]
    confidence = probs[pred_class].item()

    if pred_label == "Positive":
        st.success(f"**Pozitif** 😊 (güven: %{confidence*100:.1f})")
    else:
        st.error(f"**Negatif** 😞 (güven: %{confidence*100:.1f})")

    st.write("---")
    st.write("**Detaylı olasılıklar:**")
    col1, col2 = st.columns(2)
    col1.metric("Negative", f"%{probs[0].item()*100:.1f}")
    col2.metric("Positive", f"%{probs[1].item()*100:.1f}")

    with st.expander("Kelime bazlı tokenizasyon (nasıl işlendi?)"):
        tokens = tokenizer.tokenize(text)
        st.write(tokens)