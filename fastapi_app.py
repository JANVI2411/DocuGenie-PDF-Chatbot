from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks
import os 

from scripts.pdf_qa import ChatBotModel

app = FastAPI()
# uvicorn fastapi_app:app --reload
UPLOAD_DIRECTORY = "uploaded_pdf"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

chatbot = ChatBotModel()

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # Define the path where the file will be saved
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    # Save the uploaded PDF to the local directory
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    return {"filename": file.filename, "msg": "PDF uploaded and saved successfully!"}

@app.post("/start_pdf_processing/")
async def start_pdf_processing(filename: str = Form(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    file_location = os.path.join(UPLOAD_DIRECTORY, filename)
    background_tasks.add_task(chatbot.pdf_parser,file_location)
    background_tasks.add_task(chatbot.get_vectorstore,filename)
    return {"status":"Success","msg": "Started processing the PDF."}

@app.get("/get_pdf_status/")
def get_pdf_status():
    return {"status":chatbot.pdf_status}

@app.post("/chat_with_model/")
def chat_with_model(query: str = Form(...), pdf_name: str = Form(...)):
    response = chatbot.llm_client_chat(query,pdf_name)
    return {"status":"Success","answer": response}