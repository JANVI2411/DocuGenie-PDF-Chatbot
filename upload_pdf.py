import streamlit as st
import requests
import time 

# Initialize chat history in session state for the chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "pdf_status" not in st.session_state:
    st.session_state.pdf_status = ""
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

st.title("Upload PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Uploading PDF..."):
        API_URL = "http://127.0.0.1:8000/uploadfile/"
        file_name = uploaded_file.name
        st.session_state.pdf_name = uploaded_file.name
        file_bytes = uploaded_file.read()
        files = {
            "file": (file_name, file_bytes, "application/pdf")
        }
        response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            st.session_state.pdf_text = "Success"
            upload_msg = response.json()
            st.success("PDF uploaded.")
        else:
            st.write("Couldnt upload PDF")
    
    API_URL = "http://127.0.0.1:8000/start_pdf_processing/"
    response = requests.post(API_URL, data= { "filename" : file_name})
    if response.status_code == 200:
        with st.spinner("Processing PDF..."):
            API_URL = "http://127.0.0.1:8000/get_pdf_status/"
            response = requests.get(API_URL)

            if response.status_code == 200:
                msg = response.json()
                while msg["status"] != "Processed":
                    API_URL = "http://127.0.0.1:8000/get_pdf_status/"
                    response = requests.get(API_URL)
                    msg = response.json()
                    time.sleep(5)
                st.session_state.pdf_status = "Processed"
                st.success("PDF processed! You can now go to the Chatbot page.")
            else:
                st.write("Couldnt process PDF")
    else:
        st.write("Couldnt process PDF")
        if response: st.write(response.json())

chatbot_page = st.Page("pages/chatbot.py", title="Chatbot", icon=":material/handyman:")
