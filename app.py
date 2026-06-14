import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Page configuration
st.set_page_config(
    page_title="Türkçe Duygu Analizi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
    }
    .score-text {
        font-size: 24px;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #6c757d;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Model paths (referencing nlp_api-130 cache)
MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"
MODEL_PATH = r"C:\Users\Vektorel\Desktop\nlp_api-130\model\sentiment_model"
TOKENIZER_PATH = r"C:\Users\Vektorel\Desktop\nlp_api-130\model\tokenizer"

@st.cache_resource
def load_model():
    """Loads the model and tokenizer, using cache if available."""
    try:
        if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        return tokenizer, model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, None

# Sidebar
with st.sidebar:
    st.title("🤖 NLP Analiz")
    st.info("Bu uygulama, Türkçe metinler üzerinde duygu analizi yapmak için BERT modelini kullanır.")
    st.divider()
    st.write("**Model:** BERT Base Turkish")
    st.write("**Geliştirici:** Nejdet TUT")

# Main Page
st.title("✨ Türkçe Duygu Analizi")
st.write("Analiz etmek istediğiniz metni aşağıya girin:")

text_input = st.text_area("Metin Girişi", placeholder="Örn: Bu ürün gerçekten harika, kesinlikle tavsiye ederim...", height=150)

if text_input:
    with st.spinner("⏳ Analiz ediliyor..."):
        tokenizer, model = load_model()

        if tokenizer and model:
            inputs = tokenizer(
                text_input,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)

            neg_score = float(probs[0][0])
            pos_score = float(probs[0][1])

            sentiment = "POZİTİF" if pos_score > neg_score else "NEGATİF"
            conf = max(neg_score, pos_score)
            css_class = "positive" if sentiment == "POZİTİF" else "negative"
            icon = "✅" if sentiment == "POZİTİF" else "❌"

            st.markdown(f"""
            <div class="sentiment-box {css_class}">
                <h2>{icon} {sentiment}</h2>
                <div class="score-text">Güven Skoru: %{conf*100:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            # Detailed Scores
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pozitif Skoru", f"{pos_score:.4f}")
                st.progress(int(pos_score * 100))
            with col2:
                st.metric("Negatif Skoru", f"{neg_score:.4f}")
                st.progress(int(neg_score * 100))
else:
    st.info("Lütfen analiz için bir metin girin.")

st.markdown('<div class="footer">BERT Base Turkish Sentiment Analysis • © 2026</div>', unsafe_allow_html=True)
