import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# --- 1. Load Data and Train Model ---
@st.cache_resource 
def load_and_train_model():
    try:
        # Load your real dataset
        df = pd.read_csv("data.csv")
        
        # Features and Target
        X = df[['Funding_Access_Score', 'Employee_Satisfaction_Score', 'Market_Fit_Score']]
        y = df['Closed']
        
        # Train-Test Split (to calculate accuracy)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate Metrics
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Train on full data for final predictions
        model.fit(X, y)
        
        return model, acc, True
    except Exception as e:
        return str(e), 0, False

model, accuracy, success = load_and_train_model()

# --- 2. Chatbot Logic ---
def get_bot_response(user_input):
    user_input = user_input.lower()
    
    if "hello" in user_input or "hi" in user_input:
        return "Hello! I am your IT Firm Sustainability Advisor. Ask me about our research metrics or type 'predict' to test the real data model."
    elif "metrics" in user_input or "accuracy" in user_input:
        return f"Based on our dataset of {153} responses, our Random Forest model currently achieves an accuracy of {accuracy*100:.2f}%!"
    elif "predict" in user_input:
        return "PREDICTION_MODE"
    elif "cause" in user_input or "why" in user_input:
        return "Our research indicates that economic instability, lack of proper funding access, and poor market fit are leading causes of mass closures among BD IT firms."
    else:
        return "I am a prototype AI. Try asking about 'accuracy', 'causes of closure', or type 'predict' to check firm survival."

# --- 3. Streamlit UI ---
st.set_page_config(page_title="BD IT Firm Sustainability", page_icon="🏢")
st.title("🤖 BD IT Firm Sustainability Analyzer")
st.write("Predicting IT firm closure based on Bangladeshi stakeholder survey data.")

if not success:
    st.error(f"Data loading error! Ensure 'data.csv' is in the folder. Details: {model}")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question or type 'predict'..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_bot_response(prompt)

    if response == "PREDICTION_MODE" and success:
        with st.chat_message("assistant"):
            st.markdown("📈 **Prediction Module Activated!** Adjust the firm's metrics on the left sidebar and click the predict button.")
            
            st.sidebar.header("🏢 Input Firm Metrics")
            f_access = st.sidebar.slider("Funding Access Score", 1, 5, 3) # Adjusted scale based on data
            e_sat = st.sidebar.slider("Employee Satisfaction", 1, 10, 5)
            m_fit = st.sidebar.slider("Market Fit Score", 1, 5, 3) # Adjusted scale based on data
            
            if st.button("Run Random Forest Prediction"):
                pred = model.predict([[f_access, e_sat, m_fit]])
                if pred[0] == 0:
                    result = "🟢 SURVIVED / STABLE"
                    msg = f"**Analysis Result:** With these metrics (Funding: {f_access}, Satisfaction: {e_sat}, Market Fit: {m_fit}), the firm is predicted to be **{result}**."
                else:
                    result = "🔴 AT RISK OF CLOSURE"
                    msg = f"**Analysis Result:** With these metrics (Funding: {f_access}, Satisfaction: {e_sat}, Market Fit: {m_fit}), the firm is **{result}**."
                
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})