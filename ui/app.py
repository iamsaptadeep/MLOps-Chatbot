import streamlit as st
import requests
import time
import os


# -------------------------------------------------
# Configuration
# -------------------------------------------------

# Get API URL from environment variable or use default
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Ensure URL ends with /chat
if not API_URL.endswith("/chat"):
    API_URL = f"{API_URL}/chat"

APP_TITLE = "Customer Support AI Platform"
APP_SUBTITLE = "Automated Assistance System"


# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    layout="centered",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------
# Custom Styling (Corporate Theme)
# -------------------------------------------------
st.markdown(
    """
    <style>

    body {
        background-color: #f8f9fa;
        font-family: "Segoe UI", Arial, sans-serif;
    }

    .main-header {
        font-size: 32px;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 4px;
    }

    .sub-header {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 24px;
    }

    .confidence-box {
        padding: 8px 12px;
        border-radius: 4px;
        background-color: #e5f0ff;
        color: #1e3a8a;
        font-size: 14px;
        margin-top: 8px;
    }

    .error-box {
        padding: 10px;
        border-radius: 4px;
        background-color: #fee2e2;
        color: #991b1b;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:

    st.markdown("### System Information")

    st.markdown(
        """
        **Application:** Customer Support AI Platform  
        **Backend:** FastAPI  
        **Model:** Sentence Transformer + Classifier  
        **Deployment:** Render Cloud  
        **Monitoring:** MLflow + Evidently
        """
    )

    st.markdown("---")

    st.markdown("### Usage Guidelines")

    st.markdown(
        """
        - Ask questions related to orders, refunds, and accounts  
        - Use clear and concise language  
        - Avoid sharing sensitive personal data  
        """
    )

    st.markdown("---")

    st.markdown("### System Status")

    try:
        # Check health endpoint (root path)
        health_url = API_URL.replace("/chat", "")
        r = requests.get(health_url, timeout=5)

        if r.status_code == 200:
            st.success("✅ API Service: Online")
        else:
            st.warning("⚠️ API Service: Unstable")

    except Exception as e:
        st.error("❌ API Service: Offline")
        st.caption(f"Error: {str(e)}")


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    f'<div class="main-header">{APP_TITLE}</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="sub-header">{APP_SUBTITLE}</div>',
    unsafe_allow_html=True
)


# -------------------------------------------------
# Session State
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------------------------
# Display Chat History
# -------------------------------------------------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


# -------------------------------------------------
# API Call
# -------------------------------------------------
def query_api(user_text):

    payload = {
        "message": user_text
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        return response.json(), None


    except requests.exceptions.Timeout:
        return None, "⏱️ Request timeout. Please try again."


    except requests.exceptions.ConnectionError:
        return None, f"🔌 Unable to connect to backend service at {API_URL}"


    except requests.exceptions.HTTPError as e:
        return None, f"⚠️ Server error: {e}"


    except Exception as e:
        return None, f"❌ Unexpected error: {e}"


# -------------------------------------------------
# User Input
# -------------------------------------------------
user_input = st.chat_input("Enter your message")


# -------------------------------------------------
# Chat Logic
# -------------------------------------------------
if user_input:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)


    # Processing indicator
    with st.spinner("Processing request..."):
        time.sleep(0.3)

        result, error = query_api(user_input)


    # Handle API error
    if error:

        error_msg = f"<div class='error-box'>{error}</div>"

        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg
        })

        with st.chat_message("assistant"):
            st.markdown(error_msg, unsafe_allow_html=True)


    # Handle success
    else:

        intent = result.get("intent", "unknown")
        response = result.get("response", "")
        confidence = result.get("confidence", 0.0)

        assistant_text = f"""
        <b>Detected Intent:</b> {intent}

        <br><br>

        {response}

        <div class="confidence-box">
        Confidence Score: {confidence:.2f}
        </div>
        """

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_text
        })

        with st.chat_message("assistant"):
            st.markdown(assistant_text, unsafe_allow_html=True)


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="text-align: center; color: #6b7280; font-size: 13px;">
    Customer Support AI Platform • Powered by Machine Learning and MLOps
    </div>
    """,
    unsafe_allow_html=True
)

