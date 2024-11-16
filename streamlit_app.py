import streamlit as st
import requests
import time 
port = "8000"

# Initialize chat history in session state for the chatbot
if 'page' not in st.session_state:
    st.session_state.page = 1  # Start at page 1
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_status" not in st.session_state:
    st.session_state.pdf_status = ""

# FastAPI URLs for login and sign-up
LOGIN_API_URL = f"http://localhost:{port}/login"  # Update with your FastAPI URL
SIGNUP_API_URL = f"http://localhost:{port}/signup"  # Update with your FastAPI URL

# Function to handle login
def login(username, password):
    payload = {"username": username, "password": password}
    response = requests.post(LOGIN_API_URL, json=payload)
    if response.status_code == 200:
        return response.json()  # Return token or user data
    return None

# Function to handle sign-up
def signup(username, password):
    payload = {"username": username, "password": password}
    response = requests.post(SIGNUP_API_URL, json=payload)
    if response.status_code == 200:
        return response.json()  # Return token or user data
    return None

# Function to manage login state
def login_user(username, token):
    st.session_state["user_id"] = user_id
    st.session_state['username'] = username
    st.session_state['logged_in'] = True

# Logout function
def logout():
    st.session_state.clear()  # Clear the session

def validate_openai_key(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        try:
            client.models.list()
        except openai.AuthenticationError:
            return False
        os.environ['OPENAI_API_KEY'] = api_key
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def response_generator(res):
    for word in res.split():
        yield word + " "
        time.sleep(0.05)

def generate_html_table(dataframe):
    table_html = """
    <style>
        table {
            width: 80%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 18px;
            text-align: left;
        }
        th, td {
            padding: 12px;
            border: 1px solid #ddd;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        td {
            text-align: center;
        }
    </style>
    <table style="width:100%">
        <tr>
            <th>Name</th>
            <th>Status</th>
        </tr>
    """
    
    # Dynamically generate table rows from the dataframe
    for index, row in dataframe.iterrows():
        table_html += f"""
        <tr>
            <td>{row['name']}</td>
            <td>{row['status']}</td>
        </tr>
        """
    
    table_html += "</table>"
    
    return table_html

    # Generate the dynamic HTML table
    
def success_login():
    st.title("PDF Chatbot")
    st.header(f"Username: {st.session_state.username}")
    # Navigation buttons
    page_options = ["1. API Key Input", "2. Upload PDF", "3. Chatbot", "4. Logout"]
    page_selection = st.sidebar.radio("Go to", page_options)

    if page_selection == "1. API Key Input":
        st.session_state.page = 1
    elif page_selection == "2. Upload PDF":
        st.session_state.page = 2
    elif page_selection == "3. Chatbot":
        st.session_state.page = 3
    elif page_selection == "4. Logout":
        st.session_state.page = 4

    if st.session_state.page == 1:
        st.header("Step 1: Enter OpenAI API Key")
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if st.button("Validate API Key"):
            if validate_openai_key(api_key):
                st.success("API key is valid!")
                st.session_state.api_key = api_key
                st.session_state.api_key_valid = True
            else:
                st.error("Invalid API key. Please try again.")
    
    if st.session_state.page == 2 and st.session_state.api_key_valid:
        st.header("Step 2: Upload PDF")

        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        description = "xyz"

        if uploaded_file is not None:
            with st.spinner("Uploading PDF..."):
                API_URL = f"http://127.0.0.1:{port}/uploadfile/"
                file_name = uploaded_file.name
                st.session_state.pdf_name = uploaded_file.name
                file_bytes = uploaded_file.read()
                files = {
                    "file": (file_name, file_bytes, "application/pdf")
                }
                json_data = {
                    "user_id":st.session_state.user_id,
                    "description":description
                }
                response = requests.post(API_URL, files=files, json = json_data)
                if response.status_code == 200:
                    st.session_state.pdf_status = "Uploaded"
                    upload_msg = response.json()
                    st.success("PDF uploaded.")
                else:
                    st.write("Couldnt upload PDF")
            
            API_URL = f"http://127.0.0.1:{port}/start_pdf_processing/"
            response = requests.post(API_URL, data= { "filename" : file_name})
            if response.status_code == 200:
                with st.spinner("Processing PDF..."):
                    API_URL = f"http://127.0.0.1:{port}/get_pdf_status/"
                    response = requests.get(API_URL)

                    if response.status_code == 200:
                        msg = response.json()
                        while msg["status"] != "Processed":
                            API_URL = f"http://127.0.0.1:{port}/get_pdf_status/"
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

    if st.session_state.page == 3 and st.session_state.api_key_valid and st.session_state.pdf_status == "Processed":
        # chatbot_page = st.Page("pages/chatbot.py", title="Chatbot", icon=":material/handyman:")
        st.header("PDF Chatbot") 
        # Ensure PDF is uploaded and procssed before allowing questions
        if st.session_state.pdf_status == "Processed":     
            
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
                        if response.status_code == 200:
                            ans = response.json()["answer"]
                            response_content = st.write_stream(response_generator(ans))
                        else:
                            response_content = st.write_stream(response_generator("Network issue"))
                # Add assistant's response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_content})

        elif st.session_state.pdf_status == "Uploaded":
            st.warning("Wait till PDF is getting processed")
        elif st.session_state.pdf_status == "":
            st.warning("Please upload a PDF on the 'Upload PDF' page.")

    if st.session_state.page == 4:
        if st.button("Logout"):
            logout()

def main():
    # html_table = open("html_files/pdf_table.txt","r").read()
    # st.markdown(html_table, unsafe_allow_html=True)
    import pandas as pd
    data = {
        "name":["dcgan","cyclegan"],
        "status":["uploaded","vectore store created"]
    }
    df = pd.DataFrame(data)
    table_html  = generate_html_table(df)
    st.components.v1.html(table_html, scrolling=True)
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # If logged in, show a logged-in message
    if st.session_state['logged_in']:
        # st.success(f"Welcome {st.session_state['username']}!")
        success_login()
            # st.experimental_rerun()  # Refresh page to show login form again
    else:
        # Toggle between Login and Sign-Up
        option = st.selectbox("Choose Option", ["Sign Up","Login"])

        if option == "Login":
            # Login form
            st.title("Login Page")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                result = login(username, password)
                if result["message"]=="Success":
                    login_user(username, result['token'])  # Save session info
                    # st.experimental_rerun()  # Refresh the page to show logged-in state
                else:
                    st.error("Invalid username or password.")

        elif option == "Sign Up":
            # Sign-Up form
            st.title("Sign Up Page")
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")

            if st.button("Sign Up"):
                result = signup(new_username, new_password)
                if result["message"]=="Success":
                    st.success("Account created successfully! Please got to login page.")
                else:
                    st.error("Sign-up failed. Try a different username.")

main()

