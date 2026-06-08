import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
import os

st.set_page_config(page_title="BD IT Firm AI Advisor", page_icon="🤖", layout="wide")

# --- ইউজারের মেসেজ ডান পাশে নেওয়ার কাস্টম ডিজাইন (Custom CSS) ---
st.markdown(
    """
    <style>
    /* ইউজারের মেসেজ ডান পাশে সরাতে */
    div[data-testid="stChatMessage"]:has(svg[title="user"]) {
        flex-direction: row-reverse;
        text-align: right;
    }
    div[data-testid="stChatMessage"]:has(svg[title="user"]) .stMarkdown {
        background-color: #2b313e; /* হালকা ব্যাকগ্রাউন্ড কালার */
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 1. Setup Gemini API ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    generation_config = {
        "temperature": 0.7, 
        "max_output_tokens": 800, 
    }
    
    # মডেলের নাম আপডেট করে 'gemini-1.5-flash-latest' দেওয়া হয়েছে
    gemini_model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
        generation_config=generation_config
    )
except Exception as e:
    st.error("Gemini API Key-তে সমস্যা আছে। দয়া করে Streamlit Secrets চেক করুন।")

# --- ফাইলের নাম চেক করা ---
if os.path.exists("Data.csv"):
    FILE_NAME = "Data.csv"
elif os.path.exists("data.csv"):
    FILE_NAME = "data.csv"
else:
    FILE_NAME = None

# --- 2. Load Data & Train Model ---
@st.cache_resource
def load_data_and_model():
    if FILE_NAME is None:
        return None, None, None, False
        
    try:
        df = pd.read_csv(FILE_NAME, on_bad_lines='skip')
        df = df.dropna(subset=['Funding_Access_Score', 'Employee_Satisfaction_Score', 'Market_Fit_Score', 'Closed'])
        
        if df.empty:
            return None, None, None, False
            
        X = df[['Funding_Access_Score', 'Employee_Satisfaction_Score', 'Market_Fit_Score']]
        y = df['Closed'].astype(int)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        ind_avg = {
            'Funding': df['Funding_Access_Score'].mean(),
            'Satisfaction': df['Employee_Satisfaction_Score'].mean(),
            'Market_Fit': df['Market_Fit_Score'].mean()
        }
        return df, rf_model, ind_avg, True
    except Exception as e:
        return None, None, None, False

result = load_data_and_model()
if result[3]:
    df, rf_model, industry_avg, success = result
else:
    success = False

# --- 3. Save New Data ---
def save_new_company_data(name, f_access, e_sat, m_fit, pred):
    try:
        new_data = pd.DataFrame({
            'Organization_Name': [name],
            'Employee_Satisfaction_Score': [e_sat],
            'Funding_Access_Score': [f_access],
            'Market_Fit_Score': [m_fit],
            'Closed': [pred]
        })
        
        save_name = FILE_NAME if FILE_NAME else "data.csv"
        
        if os.path.exists(save_name):
            existing_df = pd.read_csv(save_name, on_bad_lines='skip')
            updated_df = pd.concat([existing_df, new_data], ignore_index=True)
            updated_df.to_csv(save_name, index=False)
        else:
            new_data.to_csv(save_name, index=False)
            
        load_data_and_model.clear() 
    except Exception as e:
        st.sidebar.error(f"Data saving error: {e}")

# --- 4. Smart Gemini Logic ---
def get_gemini_response(user_query, df, rf_model, industry_avg):
    user_query_lower = user_query.lower()

    context = ""
    companies = df['Organization_Name'].dropna().unique()
    for company in companies:
        if str(company).lower() in user_query_lower:
            comp_data = df[df['Organization_Name'] == company].iloc[-1] 
            f, s, m = comp_data['Funding_Access_Score'], comp_data['Employee_Satisfaction_Score'], comp_data['Market_Fit_Score']
            pred = rf_model.predict([[f, s, m]])[0]
            status = "High Risk of Closure" if pred == 1 else "Stable and Likely to Survive"
            context += f"\nData for '{company}': Funding Access={f}, Employee Satisfaction={s}, Market Fit={m}. ML Prediction={status}."

    prompt = f"""
    You are an Expert AI Business Advisor for IT firms in Bangladesh.
    Industry Averages: Funding={industry_avg['Funding']:.2f}, Satisfaction={industry_avg['Satisfaction']:.2f}, Market Fit={industry_avg['Market_Fit']:.2f}.

    Available Company Context based on user query:{context}

    User Query: "{user_query}"

    Crucial Instructions:
    1. If the user explicitly asks to "generate a report", "create a report", or "full report", provide a highly structured, formal business report (with Executive Summary, Comparative Statistics, ML Prediction analysis, Market Fit evaluation, Forecast, and Actionable Steps).
    2. If the user simply asks about stats, risk, current situation, or a general question, DO NOT generate a full report. Instead, provide a detailed, conversational, and direct answer based on the context.
    3. Make sure the output is professional, uses markdown formatting (which is easily copiable), and is entirely in English.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"API Error: {str(e)}"

# --- 5. UI Layout & Chat Interface ---
st.title("🤖 AI-Powered IT Firm Assistant & Knowledge Base")

if not success:
    st.error("⚠️ ডেটাসেট (data.csv) খুঁজে পাওয়া যাচ্ছে না অথবা ফাইলটি খালি! দয়া করে গিটহাবে আপনার 'data.csv' ফাইলটি আপলোড করুন।")
else:
    with st.sidebar:
        st.header("➕ Add Company Data")
        st.write("Input data here to save it directly to the CSV knowledge base.")
        with st.form("add_company_form"):
            c_name = st.text_input("Company Name", placeholder="e.g. NextGen IT")
            c_f = st.slider("Funding Access", 1.0, 10.0, 5.0)
            c_s = st.slider("Employee Satisfaction", 1.0, 10.0, 5.0)
            c_m = st.slider("Market Fit", 1.0, 10.0, 5.0)
            submitted = st.form_submit_button("Save Data & Update AI")

            if submitted and c_name:
                pred = rf_model.predict([[c_f, c_s, c_m]])[0]
                save_new_company_data(c_name, c_f, c_s, c_m, pred)
                st.success(f"{c_name} data saved successfully! The AI knowledge base is updated.")
                st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome! You can add new company data using the sidebar to update my knowledge base. Ask me for quick stats, risk analysis, or command me to *'generate a full report'* for any saved company. You can copy my responses easily!"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) 

    if prompt := st.chat_input("Type your question or request a report here..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing knowledge base..."):
            response = get_gemini_response(prompt, df, rf_model, industry_avg)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})