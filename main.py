from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from rag_utils import ResumeRAG

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize RAG system
rag_system = ResumeRAG()

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        rag_system.process_resume(file_path)
        return {"filename": file.filename}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        answer = rag_system.ask(question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)