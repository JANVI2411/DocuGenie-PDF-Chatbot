from fastapi import FastAPI, HTTPException,File, Form, UploadFile, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import os 
import sys 
from scripts.pdf_qa import ChatBotModel
from database.db_helper import Database

# uvicorn fastapi_app:app --reload
UPLOAD_DIRECTORY = "uploaded_pdf"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

print("Loading model...")
chatbot = ChatBotModel()
app = FastAPI()

class User(BaseModel):
    username: str
    password: str

class PDF(BaseModel):
    user_id: int
    description: str 

class ChatModel(BaseModel):
    user_id: int
    query: str

class UserModel(BaseModel):
    user_id: int 

@app.post("/signup")
def signup(user: User):
    if user.username in users_db:
        raise {"message": "Failed"}
    with Database as db:
        db.add_record("user_info",{"name":user.username,"password":user.password})
    return {"message": "Success"}

@app.post("/login")
def login(user: User):
    print("login: ",users_db)
    with Database as db:
        user_info = db.get_table_data("user_info",{"name":user.username,"password":user.password})
    if user_info:
        return {"message": "Success","user_id":user_info[0]["user_id"]}
    return {"message": "Failed"}

@app.get("/")
def get_pdf_status():
    return {"status":"success"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), pdf: PDF):
    saved_name = str(uuid.uuid4()).replace("-", "_") + ".pdf"
    file_location = os.path.join(UPLOAD_DIRECTORY, saved_name)
    
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    pdf_info={
        "user_id":pdf.user_id
        "org_name":file.filename,
        "saved_name":saved_name,
        "description":pdf.description,
        "status":"Uploaded"
    }
    with Database as db:
        db.add_record("pdf_status",pdf_info)
    
    return {"status":"success","filename": file.filename, "msg": "PDF uploaded and saved successfully!"}

@app.get("/getFiles/")
def get_pdf_files(user: UserModel):
    with Database as db:
        pdf_info = db.get_table_data("pdf_status",{"user_id":user.user_id})
    
    data = []
    for row in pdf_info:
        data.append({
            "name":row["org_name"],
            "description":row["description"],
            "status":row["status"]
        })
    return {"status":"success","msg":"Files are retrived","data":data}


@app.post("/chat_with_model/")
def chat_with_model(chat : ChatModel):
    response = chatbot.invoke_llm(chat.query,chat.user_id)
    return {"status":"Success","answer": response}

if __name__ == "__main__":
    print("Running App....")
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000)
