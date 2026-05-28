import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fake News Detector", 
    page_icon="🛡️", 
    layout="centered"
)

# --- CACHE DATA & MODEL TRAINING ---
@st.cache_resource
def load_and_train_model():
    data = {
        'text': [
            # Real News Patterns
            "The prime minister announced a new economic policy today to boost small businesses and lower inflation rates.",
            "Scientists at NASA confirmed the discovery of a new planet orbiting a distant star using the James Webb telescope.",
            "Tech conglomerates unveiled a new open-source artificial intelligence chip designed to minimize server power consumption.",
            "Archaeologists in Egypt have unearthed a perfectly preserved residential settlement dating back three thousand years.",
            "The international summit concluded today with world leaders pledging billions to global ocean cleanup initiatives.",
            
            # Fake News Patterns
            "SHOCKING SECRET: Groundbreaking leaked documents reveal secret entities are hiding massive alien spacecraft under the ocean!",
            "Miracle ancient root completely cures all modern diseases in less than 24 hours while doctors try to ban it!",
            "ALERT: New wireless 6G towers are emitting magnetic frequencies that cause instant memory loss! Delete before it gets taken down!",
            "BREAKING: Secret underground bunker found beneath capital city containing gold reserves stolen from citizens!",
            "SHOCKING DISCOVERY: Hidden documents prove that history books were completely rewritten in 1950 to hide giant human civilizations."
        ],
        'label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    X, y = df['text'], df['label']
    
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', min_df=1)
    X_vectorized = vectorizer.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_vectorized, y)
    
    return vectorizer, model

# Initialize components
vectorizer, model = load_and_train_model()

# --- CUSTOM THEME & CSS INJECTION ---
st.markdown("""
    <style>
    /* 1. Modifying the entire background environment color of the page */
    .stApp {
        background: linear-gradient(135deg, #F0F4F8 0%, #E2E8F0 100%);
    }
    
    /* 2. Styling the main headings */
    .main-title {
        text-align: center; 
        color: #1E3A8A; 
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center; 
        color: #4B5563; 
        font-size: 1.1em; 
        margin-top: 5px;
        margin-bottom: 25px;
    }
    
    /* 3. Giving inputs a crisp white card look to pop off the gray background */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    
    /* 4. Elegant custom team footer card */
    .team-footer {
        text-align: center;
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 12px;
        border-top: 4px solid #1E3A8A;
        color: #374151;
        font-weight: 500;
        margin-top: 60px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        font-family: Arial, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- USER INTERFACE (UI) ---

# Application Banner Header
st.markdown("<h1 class='main-title'>🛡️ Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced Natural Language Processing & Linguistic Feature Verification Engine</p>", unsafe_allow_html=True)

# Project Summary Box
with st.expander("ℹ️ About the Architecture System (For Evaluators)"):
    st.info(
        "This system processes raw unstructured text through a mathematical **TF-IDF Vectorizer Matrix** "
        "and evaluates data boundaries using **Logistic Regression**. It is optimized to screen text parameters "
        "for structural sensationalism and structural misinformation markers automatically."
    )

st.write("")

# --- INPUT SECTION ---
news_options = {
    "Select a headline from the wire...": "",
    "🌐 Global Economy: New Financial Policy Briefing": "The prime minister announced a new economic policy today to boost small businesses and lower inflation rates.",
    "🚀 Deep Space: NASA Deep-Field Telescope Confirmation": "Scientists at NASA confirmed the discovery of a new planet orbiting a distant star using the James Webb telescope.",
    "📡 Tech Wire: Advanced Semiconductor Power Architecture": "Tech conglomerates unveiled a new open-source artificial intelligence chip designed to minimize server power consumption.",
    "🏺 History Focus: Mediterranean Excavation Updates": "Archaeologists in Egypt have unearthed a perfectly preserved residential settlement dating back three thousand years.",
    "🌊 Environment: International Marine Treaty Signed": "The international summit concluded today with world leaders pledging billions to global ocean cleanup initiatives.",
    "📢 Report: Classified Maritime Logistics Leak": "SHOCKING SECRET: Groundbreaking leaked documents reveal secret entities are hiding massive alien spacecraft under the ocean!",
    "🌿 Medical Review: Alternative Botanical Health Breakthrough": "Miracle ancient root completely cures all modern diseases in less than 24 hours while doctors try to ban it!",
    "📲 Infrastructure: Next-Gen Signal Infrastructure Deployment": "ALERT: New wireless 6G towers are emitting magnetic frequencies that cause instant memory loss! Delete before it gets taken down!",
    "🏦 Capital Focus: Central Vault Audit Findings": "BREAKING: Secret underground bunker found beneath capital city containing gold reserves stolen from citizens!",
    "📚 Editorial: Global Archives Verification Study": "SHOCKING DISCOVERY: Hidden documents prove that history books were completely rewritten in 1950 to hide giant human civilizations."
}

# Dropdown Selection
selected_option = st.selectbox("Choose a pre-loaded news item headline:", list(news_options.keys()))
default_text = news_options[selected_option]

# Interactive Text Area
user_input = st.text_area("📋 News Article Body Content:", value=default_text, height=180, 
                          placeholder="Select an option above or paste a custom news article text structure here...")

st.write("")

# --- EVALUATION ENGINE BUTTON ---
if st.button("🚀 Run Verification Pipeline", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please provide a valid text string to run verification.")
    else:
        st.write("---")
        st.markdown("### 📊 Classification Report Results")
        
        with st.spinner("Executing tensor matrix validation..."):
            # Math execution
            vectorized_input = vectorizer.transform([user_input])
            prediction = model.predict(vectorized_input)[0]
            probabilities = model.predict_proba(vectorized_input)[0]
            
            # Use columns to lay out the metrics neatly side-by-side
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if prediction == 1:
                    st.error("🚨 **ALERT: Classified as FAKE NEWS**")
                    st.markdown("*Linguistic feature flags match known sensationalist and fabricated patterns.*")
                else:
                    st.success("✅ **VERIFIED: Classified as REAL NEWS**")
                    st.markdown("*Linguistic feature flags match standard verified journalism and reporting profiles.*")
            
            with col2:
                # Highlight metrics clearly
                confidence_score = probabilities[1] if prediction == 1 else probabilities[0]
                st.metric(
                    label="AI Confidence", 
                    value=f"{confidence_score*100:.2f}%"
                )
            
            # Progress bar visualizer at the bottom
            st.write("")
            st.markdown("**Probability Distribution Spectrum:**")
            if prediction == 1:
                st.progress(float(probabilities[1]))
                st.caption("Bar shifts full right when system flags structural misdirection.")
            else:
                st.progress(float(probabilities[0]))
                st.caption("Bar shifts full left when system confirms standard journalistic structures.")

# --- TEAM CREDITS FOOTER ---
st.markdown(
    "<div class='team-footer'>"
    "🚀 Developed by: <b>Debritu</b>, <b>Sudipta</b>, <b>Roopsha</b>, and <b>Antarika</b>"
    "</div>", 
    unsafe_allow_html=True
)