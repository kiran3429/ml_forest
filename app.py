import streamlit as st
import joblib
import requests
import io

# Google Drive File ID
FILE_ID = "13AqXvvCcmHNggKDitu-o1HBho12Mgl1N"

@st.cache_resource
def load_model_from_drive(file_id):
    try:
        # Step 1: Try to get file directly
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, allow_redirects=True)

        # Step 2: Handle cases where Drive requires confirmation
        if "quota exceeded" in response.text.lower() or "download_warning" in response.text.lower():
            confirm_token = None
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    confirm_token = value
            if confirm_token:
                url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                response = requests.get(url, allow_redirects=True)

        # Step 3: Verify response
        if response.status_code != 200:
            st.error(f"❌ Failed to fetch model from Drive (Status {response.status_code})")
            return None

        # Step 4: Load model with joblib
        model = joblib.load(io.BytesIO(response.content))
        return model

    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None


ensemble_model = load_model_from_drive(FILE_ID)
if ensemble_model:
    st.success("✅ Model loaded successfully from Google Drive!")
