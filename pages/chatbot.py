import streamlit as st
import time 
import requests

def response_generator(res):
    for word in res.split():
        yield word + " "
        time.sleep(0.05)

port = "5000"

st.title("PDF Question Answering Chatbot")
                
# Ensure PDF is uploaded before allowing questions
if st.session_state.pdf_text == "": 
    st.warning("Please upload a PDF on the 'Upload PDF' page.")
elif st.session_state.pdf_status == "":
    st.warning("Please upload a PDF again on the 'Upload PDF' page.")
else:
    # Step 2: Chatbot Interface with Persistent Chat
    # st.subheader("Ask a question about the PDF")
    # print(st.session_state.pdf_text)
    # Display chat messages from history on app rerun
    with st.spinner("Loading LLM Model..."):
        API_URL = f"http://127.0.0.1:{port}/get_model_status/"
        response = requests.get(API_URL)
        if response.status_code == 200:
            msg = response.json()
            while msg["status"] != "Loaded":
                API_URL = f"http://127.0.0.1:{port}/get_model_status/"
                response = requests.get(API_URL)
                msg = response.json()
                time.sleep(5)

    print(st.session_state.messages)
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Take user input for chatbot
    if prompt := st.chat_input("Ask your question here..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Stream the assistant's response
        response_content = ""
        with st.chat_message("assistant"):
            with st.spinner("..."):
                API_URL = f"http://127.0.0.1:{port}/chat_with_model/"
                response = requests.post(API_URL, data = {"query":prompt,"pdf_name":st.session_state.pdf_name})
                print("-->",response)
                if response.status_code == 200:
                    ans = response.json()["answer"]
                    response_content = st.write_stream(response_generator(ans))
                else:
                    response_content = st.write_stream(response_generator("Network issue"))
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})
