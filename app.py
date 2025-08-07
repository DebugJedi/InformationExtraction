from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import tempfile
from fastapi.responses import JSONResponse
from fastapi.middle
try:
    from src.config import LocalRag
    from src.utils import extract_text
except Exception as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.error(f"Failed to import modules: {e}")
    raise

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "RAG API is running. Use /docs to interact with it. "}


@app.post("/query_pdf")
async def query_from_pdf(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    input_text = extract_text.from_pdf(tmp_path)
    rag = LocalRag(raw_text=input_text)

    answer = rag.query(question)

    return JSONResponse(content = {"answer": answer})