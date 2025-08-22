from fastapi import FastAPI, UploadFile, File, HTTPException, status, Form
from fastapi.responses import JSONResponse
import tempfile, os, logging

from src.utils import extract_text

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "RAG API is running. Use /docs to interact with it. "}


@app.post("/query_pdf", summary="Upload a PDF and ask your question.")
async def query_pdf(file: UploadFile=File(..., description="PDF file for knowledgebase."),
        question: str= Form(..., description="Your question about the PDF.")):
    # Validate content type
    if file.content_type not in ("application/pdf", "appliation/octet-stream"):
        raise HTTPException(status_code=400, detail="Please upload a PDF (content-type) application/pdf).")
    
    # Save the upload to a temp file
    try: 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(data)
            pdf_path = tmp.name
    except Exception as e:
        logger.exception("Failed to persist uploaded file")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from e


    # OCR extraction
    try:
        text = extract_text.from_pdf(pdf_path)
    except Exception as e:
        logger.exception("OCR/text extraction failed!")
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF.") from e 
    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            pass

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted form the PDF.")

    try:
        from src.config import LocalRag

        rag = LocalRag(raw_text=text)
        answer = rag.query(question)
        return {"answer": answer}
    except Exception as e:
        logger.exception("RAG pipeline failed!")
        raise HTTPException(status_code=500, detail=f"Failed to run RAG query: {e}...")       
    



